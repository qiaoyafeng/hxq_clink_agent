"""ASR 抽象接口."""

from abc import ABC, abstractmethod


class ASRInterface(ABC):
    """语音识别抽象基类."""

    @abstractmethod
    async def recognize(self, pcm: bytes, sample_rate: int = 8000) -> str:
        """将 PCM 音频数据转为文本.

        Args:
            pcm: 原始 PCM 字节数据（16bit signed LE）
            sample_rate: 采样率，默认 8000Hz

        Returns:
            识别出的文本
        """
        ...
