"""会话管理 - 单通话 WebSocket 连接的生命周期管理."""

import asyncio
import json
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from .interfaces.asr_streaming import ASRStreamingInterface
from .interfaces.voice_to_voice import VoiceToVoiceInterface
from .audio_buffer import AudioBuffer
from .config import Settings
from .pipeline import Pipeline
from .protocol import (
    build_ctl_frame,
    build_dat_downlink,
    build_resource_control,
    build_session_result,
    build_transfer_frame,
    parse_binary_frame,
    parse_text_frame,
)


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
        voice_adapter: VoiceToVoiceInterface | None = None,
    ):
        self.session_id = str(uuid.uuid4())
        # 初始从 query params 提取，后续可被 |CTL|00| 中的参数覆盖
        self.unique_id = params.get("uniqueId", params.get("uuid", "unknown"))
        self.enterprise_id = params.get("enterpriseId", "")
        self.cno = params.get("cno", "")
        self.monitor_side = params.get("monitorSide", "0")

        self._ws = ws
        self._pipeline = pipeline
        self._settings = settings
        self._closed = False

        # 流式 ASR（优先使用）
        self._asr_streaming = asr_streaming

        # Voice-to-Voice 适配器（与 ASR-LLM-TTS 管线互斥）
        self._voice_adapter = voice_adapter

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

        # Voice-to-Voice relay tasks
        self._voice_relay_audio_task: asyncio.Task | None = None
        self._voice_relay_event_task: asyncio.Task | None = None
        # Voice-to-Voice: TTS 播报状态（用于 barge-in 检测）
        self._voice_tts_speaking: bool = False

        # 流式音频发送缓冲（用于累积不足一帧的残余数据）
        self._send_buffer = b""

        # 下行音频资源 key（每次新的回复递增，用于客户端排序播放）
        self._resource_key: int = 0

        # 转人工状态：True 表示已发送 |CTL|03|，等待客户端 |MSG|02| ACK
        self._transferring: bool = False
        # 转人工关键词列表（预解析）
        if settings.transfer_enabled:
            self._transfer_keywords: list[str] = [
                kw.strip()
                for kw in settings.transfer_keywords.split(",")
                if kw.strip()
            ]
        else:
            self._transfer_keywords = []

        logger.info(
            f"Session created: id={self.session_id}, "
            f"uniqueId={self.unique_id}, cno={self.cno}, "
            f"streaming={'yes' if self._asr_streaming else 'no'}, "
            f"voice_adapter={'yes' if self._voice_adapter else 'no'}"
        )

    async def run(self) -> None:
        """会话主循环：等待客户端 |CTL|00| 握手 → 接收消息并处理."""
        try:
            # 等待客户端发送 |CTL|00| 开启 session
            if not await self._wait_session_start():
                return

            # 启动流式 ASR 或 Voice-to-Voice 适配器
            if self._voice_adapter:
                # Voice-to-Voice 模式：启动适配器和 relay 协程
                await self._voice_adapter.start()
                self._voice_relay_audio_task = asyncio.create_task(
                    self._relay_audio_loop()
                )
                self._voice_relay_event_task = asyncio.create_task(
                    self._relay_event_loop()
                )
            elif self._asr_streaming:
                # 流式 ASR 模式
                await self._asr_streaming.start()
                # 启动并发 task 持续消费已识别的句子
                self._sentence_task = asyncio.create_task(self._consume_sentences())
                # 启动 barge-in 监控（仅在配置启用时）
                if self._settings.barge_in_enabled:
                    self._barge_in_task = asyncio.create_task(
                        self._monitor_barge_in()
                    )

            # 回复 session 开启结果
            await self._ws.send_text(
                build_session_result(self.unique_id)
            )

            while not self._closed:
                message = await self._ws.receive()

                if message["type"] == "websocket.receive":
                    if "bytes" in message and message["bytes"]:
                        # 二进制数据 = 帧头 + PCM 音频
                        raw = message["bytes"]
                        frame = parse_binary_frame(raw)
                        pcm_data = frame.payload
                        logger.debug(
                            f"Session {self.session_id}: [audio] "
                            f"frame={frame.frame_type} "
                            f"len={len(pcm_data)} bytes "
                            f"head={pcm_data[:16].hex()}"
                        )
                        if self._voice_adapter:
                            # Voice-to-Voice 模式：直接转发到适配器
                            self._voice_adapter.feed(pcm_data)
                        elif self._asr_streaming:
                            # 流式模式：直接转发到 DashScope
                            self._asr_streaming.feed(pcm_data)
                        elif self._audio_buffer:
                            # 回退模式：本地 VAD
                            await self._audio_buffer.feed(pcm_data)
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

    async def _wait_session_start(self) -> bool:
        """等待客户端发送 |CTL|00| 帧开启会话.

        解析 param 字段并更新会话信息。
        超时 10 秒未收到则关闭连接。

        Returns:
            True 表示握手成功，False 表示失败需退出
        """
        try:
            message = await asyncio.wait_for(
                self._ws.receive(), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Session {self.session_id}: timeout waiting for |CTL|00|, closing"
            )
            await self._ws.send_text(
                build_session_result(self.unique_id, result=1006, description="请求中断")
            )
            await self._ws.close(4000, "Session start timeout")
            return False

        if message["type"] != "websocket.receive":
            logger.warning(
                f"Session {self.session_id}: unexpected message during handshake: {message['type']}"
            )
            return False

        text = message.get("text", "")
        if not text:
            logger.warning(
                f"Session {self.session_id}: expected text |CTL|00| but got binary"
            )
            await self._ws.send_text(
                build_session_result(self.unique_id, result=1001, description="请求参数无效")
            )
            await self._ws.close(4001, "Invalid session start")
            return False

        frame = parse_text_frame(text)
        if frame.frame_type != "CTL" or frame.enum_code != "00":
            logger.warning(
                f"Session {self.session_id}: expected |CTL|00| but got: {text}"
            )
            await self._ws.send_text(
                build_session_result(self.unique_id, result=1001, description="请求参数无效")
            )
            await self._ws.close(4001, "Invalid session start")
            return False

        # 解析 CTL|00 的 payload
        try:
            ctl_data = json.loads(frame.payload) if frame.payload else {}
            param = ctl_data.get("param", {})
            # 从 param 中更新会话信息
            if param.get("uniqueId"):
                self.unique_id = param["uniqueId"]
            if param.get("enterpriseId"):
                self.enterprise_id = str(param["enterpriseId"])
            if param.get("customerNumber"):
                self.customer_number = param["customerNumber"]
            if param.get("monitorSide"):
                self.monitor_side = str(param["monitorSide"])
            if param.get("callType"):
                self.call_type = str(param["callType"])
            # 解析 userField 扩展字段
            user_field = param.get("userField", {})
            if isinstance(user_field, dict):
                self.user_field = user_field
            else:
                self.user_field = {}

            logger.info(
                f"Session {self.session_id}: CTL|00 handshake OK, "
                f"uniqueId={self.unique_id}, enterpriseId={self.enterprise_id}, "
                f"userField={self.user_field}"
            )
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                f"Session {self.session_id}: failed to parse CTL|00 payload: {e}"
            )

        return True

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
                    # 转人工关键词拦截：命中则不再进入 pipeline，直接发送 |CTL|03|
                    if self._detect_transfer(text):
                        await self._trigger_transfer(text)
                        continue
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
                    # 发送资源控制帧通知客户端中断播放
                    try:
                        await self._ws.send_text(
                            build_resource_control(self._resource_key)
                        )
                    except Exception:
                        pass
                    self._processing_task.cancel()
                    # 清空音频发送缓冲区
                    self._send_buffer = b""
                    # 递增 key 以确保后续音频不被客户端忽略
                    self._resource_key += 1
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

    # ── Voice-to-Voice relay 方法 ──

    async def _relay_audio_loop(self) -> None:
        """Voice-to-Voice 模式：持续从适配器获取音频并发送给客户端."""
        while not self._closed and self._voice_adapter:
            try:
                chunk = await self._voice_adapter.get_audio_chunk()
                if chunk is None:
                    logger.info(
                        f"Session {self.session_id}: voice adapter audio stream ended"
                    )
                    break
                if self._closed:
                    break
                await self._send_audio_chunk(chunk, self._resource_key)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Session {self.session_id}: voice relay audio error: {e}"
                )
                break

    async def _relay_event_loop(self) -> None:
        """Voice-to-Voice 模式：持续从适配器获取事件并处理."""
        while not self._closed and self._voice_adapter:
            try:
                event = await self._voice_adapter.get_event()
                if event is None:
                    logger.info(
                        f"Session {self.session_id}: voice adapter event stream ended"
                    )
                    break

                if event.event_type == "tts_begin":
                    # 新的 TTS 播报开始
                    if self._voice_tts_speaking:
                        # 上一个 TTS 未结束，视为 barge-in
                        logger.info(
                            f"Session {self.session_id}: barge-in detected "
                            f"(new TTS while previous still speaking)"
                        )
                        # 发送资源控制帧清客户端缓冲区
                        try:
                            await self._ws.send_text(
                                build_resource_control(self._resource_key)
                            )
                        except Exception:
                            pass
                        self._send_buffer = b""

                    self._resource_key += 1
                    self._send_buffer = b""
                    self._voice_tts_speaking = True
                    logger.debug(
                        f"Session {self.session_id}: TTS begin, "
                        f"resource_key={self._resource_key}"
                    )

                elif event.event_type == "tts_end":
                    # TTS 播报结束，刷出缓冲区
                    self._voice_tts_speaking = False
                    await self._flush_send_buffer(self._resource_key, is_end=True)
                    logger.debug(
                        f"Session {self.session_id}: TTS end, "
                        f"resource_key={self._resource_key}"
                    )

                elif event.event_type == "asr_text":
                    # 用户语音识别文本
                    logger.info(
                        f"Session {self.session_id}: voice ASR text: {event.content!r}"
                    )
                    # 转人工关键词检测
                    if self._detect_transfer(event.content):
                        await self._voice_adapter.interrupt()
                        await self._trigger_transfer(event.content)

                elif event.event_type == "llm_reply":
                    logger.info(
                        f"Session {self.session_id}: voice LLM reply: {event.content!r}"
                    )

                elif event.event_type == "media_ready":
                    logger.info(
                        f"Session {self.session_id}: voice media ready"
                    )

                elif event.event_type == "interrupt_word":
                    logger.info(
                        f"Session {self.session_id}: interrupt word hit: {event.content}"
                    )

                elif event.event_type == "function_call":
                    logger.info(
                        f"Session {self.session_id}: function call: {event.content}"
                    )

                elif event.event_type == "error":
                    logger.error(
                        f"Session {self.session_id}: voice adapter error: {event.content}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Session {self.session_id}: voice relay event error: {e}"
                )
                break

    async def _on_sentence_recognized(self, text: str) -> None:
        """流式 ASR 识别到完整句子后的处理（流式 LLM + TTS）."""
        logger.info(
            f"Session {self.session_id}: sentence recognized: {text!r}"
        )

        # 递增资源 key（每次新的回复对应新的 key）
        self._resource_key += 1
        current_key = self._resource_key

        # 流式管线处理：LLM 流式 → 断句 → TTS 流式 → 逐块发送
        self._send_buffer = b""
        async for pcm_chunk in self._pipeline.process_text_stream(text):
            if self._closed:
                break
            await self._send_audio_chunk(pcm_chunk, current_key)
        # 刷出缓冲区剩余数据（标记资源结束）
        await self._flush_send_buffer(current_key, is_end=True)

    async def _handle_text_message(self, message: str) -> None:
        """处理文本消息.
    
        使用统一帧解析处理天润融通协议帧：
        - |CTL|02|: 客户端挂机信令（兼容，理论上由服务端发送）
        - |CTL|03|: 转人工（服务端发送，此处仅作兼容记录）
        - |MSG|02|: 转人工 ACK（客户端回复）
        - {"action":"end"}: 向后兼容 ws_client 测试工具
        """
        frame = parse_text_frame(message)
    
        if frame.frame_type == "CTL":
            await self._handle_ctl_frame(frame.enum_code, frame.payload)
            return
    
        if frame.frame_type == "MSG":
            await self._handle_msg_frame(frame.enum_code, frame.payload)
            return
    
        # 兼容旧协议 JSON 消息
        try:
            data = json.loads(message)
            action = data.get("action", "")
    
            if action == "end":
                logger.info(f"Session {self.session_id}: received end action (legacy)")
                if self._audio_buffer:
                    await self._audio_buffer.flush()
                self._closed = True
            else:
                logger.warning(
                    f"Session {self.session_id}: unknown action: {action}"
                )
        except json.JSONDecodeError:
            logger.warning(
                f"Session {self.session_id}: unrecognized text message: {message}"
            )
    
    async def _handle_ctl_frame(self, enum_code: str, payload: str) -> None:
        """处理控制帧."""
        if enum_code == "00":
            # 已在握手阶段处理，此处记录重复收到
            logger.debug(
                f"Session {self.session_id}: duplicate CTL|00 received, ignoring"
            )
        elif enum_code == "02":
            # 挂机信令
            logger.info(f"Session {self.session_id}: CTL|02 hangup signal")
            if self._audio_buffer:
                await self._audio_buffer.flush()
            self._closed = True
        elif enum_code == "03":
            # 转人工（当前仅记录，触发逻辑后续接入）
            logger.info(
                f"Session {self.session_id}: CTL|03 transfer signal, payload={payload}"
            )
        elif enum_code == "04":
            # 资源控制帧（理论上由服务端发送，此处兼容处理）
            logger.debug(
                f"Session {self.session_id}: CTL|04 resource control, payload={payload}"
            )
        else:
            logger.debug(
                f"Session {self.session_id}: ignored CTL frame code={enum_code}"
            )
    
    async def _handle_msg_frame(self, enum_code: str, payload: str) -> None:
        """处理状态帧."""
        if enum_code == "02":
            # 转人工 ACK（客户端回复）
            if self._transferring:
                logger.info(
                    f"Session {self.session_id}: MSG|02 transfer ACK received, disconnecting"
                )
            else:
                logger.info(
                    f"Session {self.session_id}: MSG|02 received (non-transfer context)"
                )
            # 收到 ACK 后断开连接
            self._closed = True
        else:
            logger.debug(
                f"Session {self.session_id}: ignored MSG frame code={enum_code}"
            )

    def _detect_transfer(self, text: str) -> bool:
        """检测用户文本是否命中转人工关键词（子串包含）."""
        if not self._transfer_keywords:
            return False
        return any(kw in text for kw in self._transfer_keywords)

    async def _trigger_transfer(self, matched_text: str) -> None:
        """触发转人工：中断当前处理 → 发送 |CTL|03|，等待客户端 |MSG|02| ACK.

        ACK 处理在 _handle_msg_frame 中将 self._closed 置 True，
        自然退出 run 循环并进入 _cleanup（_send_hangup 会识别 _transferring 直接关闭）。
        """
        logger.info(
            f"Session {self.session_id}: transfer triggered by text: {matched_text!r}, "
            f"qno={self._settings.transfer_qno}"
        )
        self._transferring = True

        # 取消正在进行的 pipeline 处理（若有）
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()

        # 发送资源控制帧清客户端已缓存的音频（避免机器人回复与转人工冲突）
        try:
            await self._ws.send_text(build_resource_control(self._resource_key))
        except Exception:
            pass

        # 发送转人工帧 |CTL|03|{"qno":"..."}
        try:
            await self._ws.send_text(
                build_transfer_frame(self._settings.transfer_qno)
            )
        except Exception as e:
            logger.error(
                f"Session {self.session_id}: failed to send transfer frame: {e}"
            )
            self._closed = True
            return

        # 后续等待客户端 MSG|02 ACK，主循环会在收到后置 closed=True

    async def _on_speech_segment(self, pcm: bytes) -> None:
        """回退模式：VAD 检测到一段完整语音后的回调处理（流式）."""
        logger.info(
            f"Session {self.session_id}: speech segment, {len(pcm)} bytes"
        )

        # 递增资源 key
        self._resource_key += 1
        current_key = self._resource_key

        # 流式管线处理：ASR → LLM 流式 → TTS 流式 → 逐块发送
        self._send_buffer = b""
        async for pcm_chunk in self._pipeline.process_stream(pcm):
            if self._closed:
                break
            await self._send_audio_chunk(pcm_chunk, current_key)
        # 刷出缓冲区剩余数据（标记资源结束）
        await self._flush_send_buffer(current_key, is_end=True)

    async def _send_audio_frames(self, pcm: bytes, resource_key: int) -> None:
        """按帧（frame_size/frame_interval）发送 TTS PCM 数据回客户端.

        每帧添加 |DAT|01| 协议帧头。
        """
        frame_size = self._settings.pcm_frame_size
        frame_interval = self._settings.pcm_frame_interval
        offset = 0

        while offset < len(pcm) and not self._closed:
            chunk = pcm[offset : offset + frame_size]
            is_last = (offset + frame_size >= len(pcm))
            try:
                frame_data = build_dat_downlink(resource_key, is_last, chunk)
                await self._ws.send_bytes(frame_data)
            except Exception as e:
                logger.error(
                    f"Session {self.session_id}: failed to send audio frame: {e}"
                )
                break
            offset += frame_size
            if offset < len(pcm):
                await asyncio.sleep(frame_interval)

        logger.debug(
            f"Session {self.session_id}: sent {offset} bytes of TTS audio (key={resource_key})"
        )

    async def _send_audio_chunk(self, chunk: bytes, resource_key: int) -> None:
        """发送流式 TTS PCM 数据块，累积满一帧后发送.

        将收到的 PCM chunk 加入内部缓冲区，累积达到 frame_size 后
        构造带帧头的二进制消息发送给客户端。
        """
        self._send_buffer += chunk
        frame_size = self._settings.pcm_frame_size
        frame_interval = self._settings.pcm_frame_interval

        while len(self._send_buffer) >= frame_size and not self._closed:
            frame = self._send_buffer[:frame_size]
            self._send_buffer = self._send_buffer[frame_size:]
            try:
                frame_data = build_dat_downlink(resource_key, False, frame)
                await self._ws.send_bytes(frame_data)
            except Exception as e:
                logger.error(
                    f"Session {self.session_id}: failed to send audio chunk: {e}"
                )
                return
            # 帧间隔控制
            if self._send_buffer:
                await asyncio.sleep(frame_interval)

    async def _flush_send_buffer(self, resource_key: int, is_end: bool = True) -> None:
        """刷出音频发送缓冲区中的剩余数据.

        Args:
            resource_key: 当前资源 key
            is_end: 是否标记为资源结束（最后一帧）
        """
        if self._send_buffer and not self._closed:
            try:
                frame_data = build_dat_downlink(resource_key, is_end, self._send_buffer)
                await self._ws.send_bytes(frame_data)
            except Exception as e:
                logger.error(
                    f"Session {self.session_id}: failed to flush send buffer: {e}"
                )
            finally:
                self._send_buffer = b""
        elif is_end and not self._closed:
            # 缓冲区为空但需要发送结束标志，发一个空帧表示资源结束
            try:
                frame_data = build_dat_downlink(resource_key, True, b"")
                await self._ws.send_bytes(frame_data)
            except Exception:
                pass

        logger.debug(
            f"Session {self.session_id}: stream audio send complete (key={resource_key})"
        )

    async def _cleanup(self) -> None:
        """会话清理."""
        self._closed = True

        # 发送挂机信号并延迟关闭连接
        await self._send_hangup()

        # 停止 Voice-to-Voice 适配器
        if self._voice_adapter:
            # 取消 relay 协程
            for task in (self._voice_relay_audio_task, self._voice_relay_event_task):
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            await self._voice_adapter.stop()

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

        # 取消当前处理 task（ASR-LLM-TTS 管线模式）
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

    async def _send_hangup(self) -> None:
        """发送挂机信号并延迟关闭 WebSocket 连接.

        协议要求服务端发送 |CTL|02| 后 1s 关闭连接。
        转人工场景（self._transferring=True）不发送 |CTL|02|，
        直接关闭连接（|CTL|03| + |MSG|02| ACK 已完成交互）。
        """
        try:
            if self._transferring:
                await self._ws.close()
                return
            await self._ws.send_text(build_ctl_frame("02"))
            await asyncio.sleep(1.0)
            await self._ws.close()
        except Exception:
            # 连接可能已断开，忽略错误
            pass
