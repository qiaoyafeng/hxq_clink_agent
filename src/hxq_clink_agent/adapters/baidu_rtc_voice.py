"""百度RTC WebSocket 语音适配器.

实现 VoiceToVoiceInterface，通过 WebSocket 连接百度RTC大模型互动服务，
将 PCM 音频流转发到百度云端，百度完成 ASR+LLM+TTS（或端到端语音模型）后
返回 PCM 音频流。

连接流程：
1. 调用 REST API generateAIAgentCall 创建实例，获取 cid + token
2. WebSocket 连接 wss://rtc-aiotgw.exp.bcelive.com/v1/realtime?a=APPID&id=CID&t=TOKEN&ac=CODEC
3. 处理百度下发的鉴权、文本事件、二进制音频
"""

import asyncio
import json

from loguru import logger
from websockets.asyncio.client import connect

from ..interfaces.voice_to_voice import VoiceEvent, VoiceToVoiceInterface
from .audio_resampler import resample
from .baidu_rtc_client import BaiduRTCAPIClient, InstanceInfo, build_agent_config


class BaiduRTCVoiceAdapter(VoiceToVoiceInterface):
    """百度RTC大模型互动 WebSocket 语音适配器.

    每个 Session 创建一个实例，维护一条到百度RTC的 WebSocket 长连接。
    上行：8kHz PCM → 重采样到目标采样率 → WebSocket binary
    下行：WebSocket binary → 重采样到8kHz → 音频队列
    事件：WebSocket text → 解析 → 事件队列
    """

    def __init__(
        self,
        app_id: str,
        ak: str,
        sk: str,
        license_key: str,
        device_id: str = "hxq_clink_agent",
        user_id: str = "default_user",
        ws_endpoint: str = "wss://rtc-aiotgw.exp.bcelive.com/v1/realtime",
        api_endpoint: str = "rtc-aiagent.baidubce.com",
        audio_codec: str = "raw16k",
        e2e_enabled: bool = False,
        e2e_prompt: str = "",
        e2e_vcn: int = 8003,
        scene_role_name: str = "",
        scene_role_prompt: str = "",
        tts_vcn: str = "",
        tts_sayhi: str = "",
        lang: str = "zh",
        disable_auto_interrupt: bool = False,
        asr_vad: int = 200,
        client_sample_rate: int = 8000,
    ):
        self._app_id = app_id
        self._license_key = license_key
        self._device_id = device_id
        self._user_id = user_id
        self._ws_endpoint = ws_endpoint
        self._audio_codec = audio_codec
        self._client_sample_rate = client_sample_rate

        # 百度侧音频采样率（根据编码推断）
        if audio_codec == "raw16k":
            self._cloud_sample_rate = 16000
        elif audio_codec == "opus_cbr_8000":
            self._cloud_sample_rate = 8000
        elif audio_codec == "opus_cbr_24000":
            self._cloud_sample_rate = 24000
        elif audio_codec == "opus_cbr_48000":
            self._cloud_sample_rate = 48000
        else:
            self._cloud_sample_rate = 16000  # 默认16k

        # 端到端模式固定24kHz
        if e2e_enabled:
            self._cloud_sample_rate = 24000

        # 构建config
        self._config = build_agent_config(
            e2e_enabled=e2e_enabled,
            e2e_prompt=e2e_prompt,
            e2e_vcn=e2e_vcn,
            scene_role_name=scene_role_name,
            scene_role_prompt=scene_role_prompt,
            tts_vcn=tts_vcn,
            tts_sayhi=tts_sayhi,
            lang=lang,
            disable_auto_interrupt=disable_auto_interrupt,
            asr_vad=asr_vad,
            audio_codec=audio_codec,
            user_id=user_id,
        )

        # REST API 客户端
        self._api_client = BaiduRTCAPIClient(
            app_id=app_id,
            ak=ak,
            sk=sk,
            api_endpoint=api_endpoint,
        )

        # 运行时状态
        self._ws = None
        self._instance: InstanceInfo | None = None
        self._started = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # 音频和事件队列
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._event_queue: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()
        self._send_queue: asyncio.Queue[bytes | str | None] = asyncio.Queue()

        # 后台任务
        self._recv_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None

    async def start(self) -> None:
        """启动语音对话会话."""
        if self._started:
            logger.warning("BaiduRTC voice adapter: already started")
            return

        self._loop = asyncio.get_running_loop()

        # 1. 创建智能体实例
        self._instance = await self._loop.run_in_executor(
            None, self._api_client.generate_agent_call, self._config
        )

        # 2. 构建 WebSocket URL 并连接
        # 注意：id 必须使用 ai_agent_instance_id（实例ID），而非 context.cid；
        # 若误用 cid，百度会在握手后立即以 1006 断开且不下发 [E]:[LIC]:[MUST]。
        ws_url = (
            f"{self._ws_endpoint}"
            f"?a={self._app_id}"
            f"&id={self._instance.instance_id}"
            f"&t={self._instance.token}"
            f"&ac={self._audio_codec}"
        )
        logger.info(f"BaiduRTC: connecting to {ws_url}")

        self._ws = await connect(ws_url)
        self._started = True

        # 3. 启动后台发送和接收协程
        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

        logger.info(
            f"BaiduRTC voice adapter started: instance={self._instance.instance_id}, "
            f"cloud_sample_rate={self._cloud_sample_rate}, "
            f"client_sample_rate={self._client_sample_rate}, "
            f"codec={self._audio_codec}"
        )

    def feed(self, pcm: bytes) -> None:
        """推送上行 PCM 音频帧.

        接收 8kHz PCM，重采样到百度侧采样率后通过发送队列异步发送。

        Args:
            pcm: 原始 PCM 字节（16bit signed LE, 8kHz）
        """
        if not self._started:
            return

        # 重采样到百度侧采样率
        if self._client_sample_rate != self._cloud_sample_rate:
            pcm = resample(pcm, self._client_sample_rate, self._cloud_sample_rate)

        self._send_queue.put_nowait(pcm)

    async def get_audio_chunk(self) -> bytes | None:
        """异步获取下行 PCM 音频块（已重采样为8kHz）."""
        return await self._audio_queue.get()

    async def get_event(self) -> VoiceEvent | None:
        """异步获取文本事件."""
        return await self._event_queue.get()

    async def interrupt(self) -> None:
        """打断当前 TTS 播报."""
        if not self._instance:
            return
        await self._loop.run_in_executor(
            None, self._api_client.interrupt_tts, self._instance.instance_id
        )

    async def stop(self) -> None:
        """停止会话并释放资源."""
        if not self._started:
            return

        self._started = False

        # 投递 None 让消费者和发送循环退出
        await self._send_queue.put(None)
        await self._audio_queue.put(None)
        await self._event_queue.put(None)

        # 取消后台发送任务
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass

        # 取消后台接收任务
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass

        # 关闭 WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug(f"BaiduRTC: ws close error: {e}")
            self._ws = None

        # 停止智能体实例
        if self._instance:
            await self._loop.run_in_executor(
                None, self._api_client.stop_agent_instance, self._instance.instance_id
            )
            self._instance = None

        logger.info("BaiduRTC voice adapter stopped")

    async def _recv_loop(self) -> None:
        """后台接收循环：处理百度下发的文本和二进制消息."""
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    await self._handle_binary(message)
                elif isinstance(message, str):
                    await self._handle_text(message)
        except Exception as e:
            if self._started:
                logger.error(f"BaiduRTC recv loop error: {e}")
                await self._event_queue.put(
                    VoiceEvent("error", f"recv loop error: {e}")
                )
                await self._audio_queue.put(None)
                await self._event_queue.put(None)

    async def _handle_binary(self, data: bytes) -> None:
        """处理百度下发的二进制音频数据.

        重采样到客户端采样率后投递到音频队列。
        """
        if len(data) == 0:
            return

        # 重采样到客户端采样率
        if self._cloud_sample_rate != self._client_sample_rate:
            data = resample(data, self._cloud_sample_rate, self._client_sample_rate)

        await self._audio_queue.put(data)

    async def _handle_text(self, message: str) -> None:
        """处理百度下发的文本消息.

        解析事件格式并投递到事件队列。
        """
        logger.debug(f"BaiduRTC RX text: {message}")

        # 鉴权请求
        if message.startswith("[E]:[LIC]:[MUST]"):
            await self._send_license_activation()
            return

        # 媒体就绪
        if message.startswith("[E]:[MEDIA]:[READY]"):
            logger.info("BaiduRTC: media ready")
            await self._event_queue.put(VoiceEvent("media_ready"))
            return

        # 用户语音识别文本
        if message.startswith("[Q]:"):
            # [Q]:[M]: 开头的是音乐相关，跳过
            if not message.startswith("[Q]:[M]:"):
                text = message[4:]
                logger.info(f"BaiduRTC ASR: {text!r}")
                await self._event_queue.put(
                    VoiceEvent("asr_text", text)
                )
            return

        # AI回复文本
        if message.startswith("[A]:"):
            text = message[4:]
            logger.info(f"BaiduRTC LLM reply: {text!r}")
            await self._event_queue.put(
                VoiceEvent("llm_reply", text)
            )
            return

        # TTS 开始播报
        if message.startswith("[E]:[TTS_BEGIN_SPEAKING]"):
            logger.debug("BaiduRTC: TTS begin speaking")
            await self._event_queue.put(VoiceEvent("tts_begin"))
            return

        # TTS 结束播报
        if message.startswith("[E]:[TTS_END_SPEAKING]"):
            logger.debug("BaiduRTC: TTS end speaking")
            await self._event_queue.put(VoiceEvent("tts_end"))
            return

        # 打断词命中
        if message.startswith("[E]:[INT_WORD_HIT]:"):
            word = message[len("[E]:[INT_WORD_HIT]:"):]
            logger.info(f"BaiduRTC: interrupt word hit: {word}")
            await self._event_queue.put(
                VoiceEvent("interrupt_word", word)
            )
            return

        # 函数调用
        if message.startswith("[F]:[C]:"):
            json_str = message[8:]
            logger.info(f"BaiduRTC: function call: {json_str}")
            await self._event_queue.put(
                VoiceEvent("function_call", json_str)
            )
            return

        # 其他事件
        logger.debug(f"BaiduRTC: unhandled event: {message}")

    async def _send_loop(self) -> None:
        """后台发送循环：从队列取出数据并发送到百度RTC."""
        try:
            while self._started:
                data = await self._send_queue.get()
                if data is None:
                    break
                if self._ws:
                    await self._ws.send(data)
        except Exception as e:
            if self._started:
                logger.error(f"BaiduRTC send loop error: {e}")

    async def _send_license_activation(self) -> None:
        """发送 License 激活消息."""
        lic_msg = (
            f'[E]:[LIC]:[ACTIVE]:{{"devId":"{self._device_id}",'
            f'"uId":"{self._user_id}",'
            f'"licKey":"{self._license_key}"}}'
        )
        logger.info("BaiduRTC: sending license activation")
        self._send_queue.put_nowait(lic_msg)
