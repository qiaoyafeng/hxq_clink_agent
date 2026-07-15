"""TTS 抽象接口."""

from abc import ABC, abstractmethod


class TTSInterface(ABC):
    """语音合成抽象基类."""

    @abstractmethod
    async def synthesize(self, text: str, sample_rate: int = 8000) -> bytes:
        """将文本合成为 PCM 音频数据.

        Args:
            text: 待合成的文本
            sample_rate: 目标采样率，默认 8000Hz

        Returns:
            PCM 字节数据（16bit signed LE）
        """
        ...
