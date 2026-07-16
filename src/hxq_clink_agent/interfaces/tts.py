"""TTS 抽象接口."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator


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

    @abstractmethod
    async def synthesize_stream(
        self, text: str, sample_rate: int = 8000
    ) -> AsyncGenerator[bytes, None]:
        """流式将文本合成为 PCM 音频，逐块 yield.

        Args:
            text: 待合成的文本
            sample_rate: 目标采样率，默认 8000Hz

        Yields:
            PCM 字节数据块（16bit signed LE）
        """
        ...
        # 使其成为 async generator
        yield  # type: ignore  # pragma: no cover
