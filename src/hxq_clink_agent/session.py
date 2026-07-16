"""会话管理 - 单通话 WebSocket 连接的生命周期管理."""

import asyncio
import json
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from .adapters.asr_dashscope_streaming import ASRStreamingDashScope
from .audio_buffer import AudioBuffer
from .config import Settings
from .pipeline import Pipeline


class Session:
    """单通话会话，管理一个 WebSocket 连接的完整生命周期.

    职责：
    - 接收客户端推送的 PCM binary 数据
    - 流式模式：将 PCM 帧直接转发到 DashScope 流式 ASR，由服务端 VAD 断句
    - 回退模式：通过 AudioBuffer + 本地 VAD 切割语音段
    - 将识别文本送入 Pipeline（LLM → TTS）处理
    - 将 TTS 结果按帧回传给客户端
    """

    def __init__(
        self,
        ws: WebSocket,
        pipeline: Pipeline,
        settings: Settings,
        params: dict[str, str],
        asr_streaming: ASRStreamingDashScope | None = None,
    ):
        self.session_id = str(uuid.uuid4())
        self.unique_id = params.get("uniqueId", "unknown")
        self.enterprise_id = params.get("enterpriseId", "")
        self.cno = params.get("cno", "")
        self.monitor_side = params.get("monitorSide", "0")

        self._ws = ws
        self._pipeline = pipeline
        self._settings = settings
        self._closed = False

        # 流式 ASR（优先使用）
        self._asr_streaming = asr_streaming

        # 回退模式：本地 VAD 音频缓冲（仅在非流式模式下使用）
        self._audio_buffer: AudioBuffer | None = None
        if self._asr_streaming is None:
            self._audio_buffer = AudioBuffer(
                sample_rate=settings.pcm_sample_rate,
                sample_width=settings.pcm_sample_width,
                silence_sec=settings.vad_silence_sec,
                energy_threshold=settings.vad_energy_threshold,
                on_speech=self._on_speech_segment,
            )

        # 流式 ASR 句子消费 task
        self._sentence_task: asyncio.Task | None = None

        # 流式音频发送缓冲（用于累积不足一帧的残余数据）
        self._send_buffer = b""

        logger.info(
            f"Session created: id={self.session_id}, "
            f"uniqueId={self.unique_id}, cno={self.cno}, "
            f"streaming={'yes' if self._asr_streaming else 'no'}"
        )

    async def run(self) -> None:
        """会话主循环：接收消息并处理."""
        try:
            # 启动流式 ASR
            if self._asr_streaming:
                await self._asr_streaming.start()
                # 启动并发 task 持续消费已识别的句子
                self._sentence_task = asyncio.create_task(self._consume_sentences())

            # 发送 started 事件
            await self._ws.send_text(
                json.dumps({"event": "started", "sessionId": self.session_id})
            )

            while not self._closed:
                message = await self._ws.receive()

                if message["type"] == "websocket.receive":
                    if "bytes" in message and message["bytes"]:
                        # 二进制数据 = PCM 音频
                        if self._asr_streaming:
                            # 流式模式：直接转发到 DashScope
                            self._asr_streaming.feed(message["bytes"])
                        elif self._audio_buffer:
                            # 回退模式：本地 VAD
                            await self._audio_buffer.feed(message["bytes"])
                    elif "text" in message and message["text"]:
                        # 文本消息
                        await self._handle_text_message(message["text"])
                elif message["type"] == "websocket.disconnect":
                    break

        except WebSocketDisconnect:
            logger.info(f"Session {self.session_id}: client disconnected")
        except Exception as e:
            logger.error(f"Session {self.session_id}: error in run loop: {e}")
        finally:
            await self._cleanup()

    async def _consume_sentences(self) -> None:
        """持续从流式 ASR 获取完整句子并触发 Pipeline 处理."""
        while not self._closed and self._asr_streaming:
            try:
                text = await self._asr_streaming.get_sentence()
                if text is None:
                    # 识别结束或出错
                    break
                if text.strip() and not self._closed:
                    await self._on_sentence_recognized(text)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session {self.session_id}: sentence consumer error: {e}")

    async def _on_sentence_recognized(self, text: str) -> None:
        """流式 ASR 识别到完整句子后的处理（流式 LLM + TTS）."""
        logger.info(
            f"Session {self.session_id}: sentence recognized: {text!r}"
        )

        # 流式管线处理：LLM 流式 → 断句 → TTS 流式 → 逐块发送
        self._send_buffer = b""
        async for pcm_chunk in self._pipeline.process_text_stream(text):
            if self._closed:
                break
            await self._send_audio_chunk(pcm_chunk)
        # 刷出缓冲区剩余数据
        await self._flush_send_buffer()

    async def _handle_text_message(self, message: str) -> None:
        """处理文本消息."""
        try:
            data = json.loads(message)
            action = data.get("action", "")

            if action == "end":
                logger.info(f"Session {self.session_id}: received end action")
                # 回退模式：刷出缓冲区剩余音频
                if self._audio_buffer:
                    await self._audio_buffer.flush()
                self._closed = True
            else:
                logger.warning(
                    f"Session {self.session_id}: unknown action: {action}"
                )
        except json.JSONDecodeError:
            logger.warning(
                f"Session {self.session_id}: invalid JSON message: {message}"
            )

    async def _on_speech_segment(self, pcm: bytes) -> None:
        """回退模式：VAD 检测到一段完整语音后的回调处理（流式）."""
        logger.info(
            f"Session {self.session_id}: speech segment, {len(pcm)} bytes"
        )

        # 流式管线处理：ASR → LLM 流式 → TTS 流式 → 逐块发送
        self._send_buffer = b""
        async for pcm_chunk in self._pipeline.process_stream(pcm):
            if self._closed:
                break
            await self._send_audio_chunk(pcm_chunk)
        # 刷出缓冲区剩余数据
        await self._flush_send_buffer()

    async def _send_audio_frames(self, pcm: bytes) -> None:
        """按帧（frame_size/frame_interval）发送 TTS PCM 数据回客户端."""
        frame_size = self._settings.pcm_frame_size
        frame_interval = self._settings.pcm_frame_interval
        offset = 0

        while offset < len(pcm) and not self._closed:
            chunk = pcm[offset : offset + frame_size]
            try:
                await self._ws.send_bytes(chunk)
            except Exception as e:
                logger.error(
                    f"Session {self.session_id}: failed to send audio frame: {e}"
                )
                break
            offset += frame_size
            if offset < len(pcm):
                await asyncio.sleep(frame_interval)

        logger.debug(
            f"Session {self.session_id}: sent {offset} bytes of TTS audio"
        )

    async def _send_audio_chunk(self, chunk: bytes) -> None:
        """发送流式 TTS PCM 数据块，累积满一帧后发送.

        将收到的 PCM chunk 加入内部缓冲区，累积达到 frame_size 后
        发送给客户端，不足一帧的部分继续缓冲。
        """
        self._send_buffer += chunk
        frame_size = self._settings.pcm_frame_size
        frame_interval = self._settings.pcm_frame_interval

        while len(self._send_buffer) >= frame_size and not self._closed:
            frame = self._send_buffer[:frame_size]
            self._send_buffer = self._send_buffer[frame_size:]
            try:
                await self._ws.send_bytes(frame)
            except Exception as e:
                logger.error(
                    f"Session {self.session_id}: failed to send audio chunk: {e}"
                )
                return
            # 帧间隔控制
            if self._send_buffer:
                await asyncio.sleep(frame_interval)

    async def _flush_send_buffer(self) -> None:
        """刷出音频发送缓冲区中的剩余数据."""
        if self._send_buffer and not self._closed:
            try:
                await self._ws.send_bytes(self._send_buffer)
            except Exception as e:
                logger.error(
                    f"Session {self.session_id}: failed to flush send buffer: {e}"
                )
            finally:
                self._send_buffer = b""

        logger.debug(
            f"Session {self.session_id}: stream audio send complete"
        )

    async def _cleanup(self) -> None:
        """会话清理."""
        self._closed = True

        # 停止流式 ASR
        if self._asr_streaming:
            await self._asr_streaming.stop()

        # 取消句子消费 task
        if self._sentence_task and not self._sentence_task.done():
            self._sentence_task.cancel()
            try:
                await self._sentence_task
            except asyncio.CancelledError:
                pass

        # 重置回退模式缓冲区
        if self._audio_buffer:
            self._audio_buffer.reset()

        self._pipeline.clear_history()
        logger.info(f"Session {self.session_id}: cleaned up")
