"""会话管理 - 单通话 WebSocket 连接的生命周期管理."""

import asyncio
import json
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from .interfaces.asr_streaming import ASRStreamingInterface
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
        asr_streaming: ASRStreamingInterface | None = None,
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

        # Barge-in: 当前正在执行的 pipeline 处理 task
        self._processing_task: asyncio.Task | None = None
        # Barge-in: 监控用户语音活动的 task
        self._barge_in_task: asyncio.Task | None = None

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
                # 启动 barge-in 监控（仅在配置启用时）
                if self._settings.barge_in_enabled:
                    self._barge_in_task = asyncio.create_task(
                        self._monitor_barge_in()
                    )

            # 发送 started 事件
            await self._ws.send_text(
                json.dumps({"event": "started", "sessionId": self.session_id})
            )

            while not self._closed:
                message = await self._ws.receive()

                if message["type"] == "websocket.receive":
                    if "bytes" in message and message["bytes"]:
                        # 二进制数据 = PCM 音频
                        audio_data = message["bytes"]
                        logger.debug(
                            f"Session {self.session_id}: [audio] "
                            f"len={len(audio_data)} bytes "
                            f"head={audio_data[:16].hex()}"
                        )
                        if self._asr_streaming:
                            # 流式模式：直接转发到 DashScope
                            self._asr_streaming.feed(audio_data)
                        elif self._audio_buffer:
                            # 回退模式：本地 VAD
                            await self._audio_buffer.feed(audio_data)
                    elif "text" in message and message["text"]:
                        # 文本消息
                        logger.info(
                            f"Session {self.session_id}: [text] {message['text']}"
                        )
                        await self._handle_text_message(message["text"])
                elif message["type"] == "websocket.disconnect":
                    logger.info(f"Session {self.session_id}: [disconnect]")
                    break
                else:
                    logger.warning(
                        f"Session {self.session_id}: unknown message type: {message['type']}"
                    )

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
                    # 识别结束或出错：主动结束会话，避免继续空转
                    logger.warning(
                        f"Session {self.session_id}: ASR stream ended, closing session"
                    )
                    await self._notify_client_error("asr_stream_ended")
                    self._closed = True
                    break
                if text.strip() and not self._closed:
                    # 将 pipeline 处理包装为可取消的 task（支持 barge-in）
                    self._processing_task = asyncio.create_task(
                        self._on_sentence_recognized(text)
                    )
                    try:
                        await self._processing_task
                    except asyncio.CancelledError:
                        logger.info(
                            f"Session {self.session_id}: processing cancelled "
                            f"by barge-in, sentence: {text!r}"
                        )
                        # 清理未完成的对话历史
                        self._pipeline.pop_last_user_message()
                    finally:
                        self._processing_task = None
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session {self.session_id}: sentence consumer error: {e}")

    async def _monitor_barge_in(self) -> None:
        """Barge-in 监控：检测用户语音活动并取消当前处理."""
        while not self._closed and self._asr_streaming:
            try:
                await self._asr_streaming.wait_voice_activity()

                # 检查是否有正在进行的 pipeline 处理
                if (
                    self._processing_task
                    and not self._processing_task.done()
                ):
                    logger.info(
                        f"Session {self.session_id}: barge-in triggered, "
                        f"cancelling current LLM/TTS processing"
                    )
                    self._processing_task.cancel()
                    # 清空音频发送缓冲区
                    self._send_buffer = b""
                    # 清除 voice activity 状态以避免重复触发
                    self._asr_streaming.clear_voice_activity()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Session {self.session_id}: barge-in monitor error: {e}"
                )

    async def _notify_client_error(self, code: str) -> None:
        """给客户端发送一条错误事件，忽略发送失败."""
        try:
            await self._ws.send_text(
                json.dumps({"event": "error", "code": code})
            )
        except Exception:
            pass

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
        """处理文本消息.

        兼容两类文本帧：
        1. 内部 JSON 协议：{"action": "end"} —— 由 ws_client 等推流侧发送
        2. 天润融通 clink 信令协议：|CTL|nn|<payload>
           - |CTL|00|<json>: 呼叫开始信令，仅记录参数
           - |CTL|02|:       呼叫结束信令，等价于收到 end action
        """
        # 优先识别天润融通 CTL 控制帧
        if message.startswith("|CTL|"):
            await self._handle_ctl_frame(message)
            return

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

    async def _handle_ctl_frame(self, message: str) -> None:
        """解析天润融通 |CTL|nn|<payload> 控制帧."""
        # 拆分为 ["", "CTL", "nn", "<payload...>"]；payload 中可能含 '|'，故最多切 3 段
        parts = message.split("|", 3)
        if len(parts) < 3:
            logger.warning(
                f"Session {self.session_id}: malformed CTL frame: {message}"
            )
            return
        ctl_code = parts[2]
        payload = parts[3] if len(parts) > 3 else ""

        if ctl_code == "00":
            # 呼叫开始信令：记录参数用于排障
            logger.info(
                f"Session {self.session_id}: CTL start signal, payload={payload}"
            )
        elif ctl_code == "02":
            # 呼叫结束信令：等价于 end action，触发会话优雅关闭
            logger.info(f"Session {self.session_id}: CTL end signal")
            if self._audio_buffer:
                await self._audio_buffer.flush()
            self._closed = True
        else:
            logger.debug(
                f"Session {self.session_id}: ignored CTL frame code={ctl_code}"
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

        # 取消 barge-in 监控 task
        if self._barge_in_task and not self._barge_in_task.done():
            self._barge_in_task.cancel()
            try:
                await self._barge_in_task
            except asyncio.CancelledError:
                pass

        # 取消当前处理 task
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

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
