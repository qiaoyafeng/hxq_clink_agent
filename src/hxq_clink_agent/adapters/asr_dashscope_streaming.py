"""流式 ASR 适配器 - 通过 DashScope SDK 实时流式调用阿里云 Paraformer.

每个会话创建一个实例，维护到 DashScope 的长连接 WebSocket。
持续推送 PCM 帧，服务端 VAD 检测到句子结束时通过 asyncio.Queue 产出识别文本。
"""

import asyncio

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from loguru import logger

from ..interfaces.asr_streaming import ASRStreamingInterface


class _StreamingCallback(RecognitionCallback):
    """DashScope 流式识别回调，将 sentence_end 事件桥接到 asyncio Queue."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        owner: "ASRStreamingDashScope",
        voice_activity_event: asyncio.Event,
    ):
        self._loop = loop
        self._queue = queue
        self._owner = owner
        self._voice_activity_event = voice_activity_event

    def on_open(self) -> None:
        logger.debug("ASR streaming: connection opened")

    def on_close(self) -> None:
        logger.debug("ASR streaming: connection closed")

    def on_complete(self) -> None:
        logger.debug("ASR streaming: recognition completed")
        # 投递 None 表示识别结束
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def on_error(self, result: RecognitionResult) -> None:
        logger.error(
            f"ASR streaming error: request_id={result.request_id}, "
            f"message={result.message}"
        )
        # 标记已出错：服务端已断链，后续 stop() 无需再调用 SDK
        self._owner._errored = True
        # 投递 None 让等待方退出
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if not sentence:
            return

        text = sentence.get("text", "")
        if RecognitionResult.is_sentence_end(sentence):
            logger.info(f"ASR streaming sentence end: {text!r}")
            if text.strip():
                self._loop.call_soon_threadsafe(self._queue.put_nowait, text.strip())
        else:
            logger.debug(f"ASR streaming partial: {text!r}")
            # 非 sentence_end 的 partial result 表示用户正在说话，
            # 设置 voice activity event 用于 barge-in 检测
            if text.strip():
                self._loop.call_soon_threadsafe(
                    self._voice_activity_event.set
                )


class ASRStreamingDashScope(ASRStreamingInterface):
    """DashScope Paraformer 流式 ASR 适配器.

    每个 Session 创建一个实例，维护一个到 DashScope 的长连接。
    通过 feed() 持续推送 PCM 帧，服务端 VAD 检测到句子结束时
    通过 asyncio.Queue 产出识别文本。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "paraformer-realtime-8k-v2",
        sample_rate: int = 8000,
        max_sentence_silence: int = 800,
    ):
        self._api_key = api_key
        self._model = model
        self._sample_rate = sample_rate
        self._max_sentence_silence = max_sentence_silence

        self._recognition: Recognition | None = None
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._started = False
        self._errored = False
        self._loop: asyncio.AbstractEventLoop | None = None
        # Barge-in: 用户语音活动事件
        self._voice_activity_event: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        """启动流式识别连接."""
        if self._started:
            logger.warning("ASR streaming: already started")
            return

        # 设置 DashScope SDK API Key
        dashscope.api_key = self._api_key

        self._loop = asyncio.get_running_loop()
        callback = _StreamingCallback(
            self._loop, self._queue, self, self._voice_activity_event
        )

        self._recognition = Recognition(
            model=self._model,
            format="pcm",
            sample_rate=self._sample_rate,
            callback=callback,
            max_sentence_silence=self._max_sentence_silence,
            heartbeat=True,
        )

        # start() 是非阻塞的，在后台线程建立 WebSocket 连接
        await self._loop.run_in_executor(None, self._recognition.start)
        self._started = True
        logger.info(
            f"ASR streaming started: model={self._model}, "
            f"sample_rate={self._sample_rate}, "
            f"max_sentence_silence={self._max_sentence_silence}ms"
        )

    def feed(self, data: bytes) -> None:
        """推送 PCM 音频帧到 DashScope.

        此方法是线程安全的，可从 asyncio 事件循环直接调用。

        Args:
            data: 原始 PCM 字节（16bit signed LE）
        """
        if not self._started or not self._recognition:
            logger.warning("ASR streaming: feed called but not started")
            return
        self._recognition.send_audio_frame(data)

    async def get_sentence(self) -> str | None:
        """异步等待下一个完整句子.

        Returns:
            识别完成的文本；若识别结束或出错则返回 None
        """
        return await self._queue.get()

    async def wait_voice_activity(self) -> None:
        """异步等待用户语音活动信号（ASR partial result）.

        等待 event 被 set 后自动 clear，可反复调用。
        """
        await self._voice_activity_event.wait()
        self._voice_activity_event.clear()

    def clear_voice_activity(self) -> None:
        """手动清除 voice activity 状态（用于 barge-in 处理后重置）."""
        self._voice_activity_event.clear()

    async def stop(self) -> None:
        """停止识别并关闭连接."""
        if not self._started or not self._recognition:
            return

        self._started = False

        # 若服务端已出错断链，SDK 侧连接已关闭，无需再调用 stop()
        # 否则会抛出 "Speech recognition has stopped." 的冗余错误
        if self._errored:
            self._recognition = None
            logger.info("ASR streaming stopped (already closed by server)")
            return

        loop = asyncio.get_running_loop()
        # stop() 是阻塞的，放到线程池执行
        try:
            await loop.run_in_executor(None, self._recognition.stop)
        except Exception as e:
            logger.error(f"ASR streaming stop error: {e}")

        self._recognition = None
        logger.info("ASR streaming stopped")
