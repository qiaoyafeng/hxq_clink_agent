"""流式 ASR 抽象接口."""

from abc import ABC, abstractmethod


class ASRStreamingInterface(ABC):
    """流式语音识别抽象基类.

    定义实时流式 ASR 的统一接口：持续推送 PCM 帧，
    由服务端 VAD 断句后异步产出识别文本。
    支持 barge-in（语音打断）场景的 voice activity 检测。
    """

    @abstractmethod
    async def start(self) -> None:
        """启动流式识别连接."""
        ...

    @abstractmethod
    def feed(self, data: bytes) -> None:
        """推送 PCM 音频帧.

        Args:
            data: 原始 PCM 字节（16bit signed LE）
        """
        ...

    @abstractmethod
    async def get_sentence(self) -> str | None:
        """异步等待下一个完整句子.

        Returns:
            识别完成的文本；若识别结束或出错则返回 None
        """
        ...

    @abstractmethod
    async def wait_voice_activity(self) -> None:
        """异步等待用户语音活动信号（用于 barge-in 检测）."""
        ...

    @abstractmethod
    def clear_voice_activity(self) -> None:
        """手动清除 voice activity 状态."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """停止识别并关闭连接."""
        ...
