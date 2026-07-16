"""TTS 占位适配器 - 开发/测试用."""

import struct
from collections.abc import AsyncGenerator

from ..interfaces.tts import TTSInterface


class TTSStub(TTSInterface):
    """TTS 占位实现，生成静音 PCM 数据，用于开发联调."""

    async def synthesize(self, text: str, sample_rate: int = 8000) -> bytes:
        """生成与文本长度对应的静音 PCM（约每字 0.3 秒）."""
        # 每个字符约 0.3 秒的静音音频
        duration_sec = max(len(text) * 0.3, 0.5)
        num_samples = int(sample_rate * duration_sec)
        # 16bit signed LE 静音 = 全零
        return struct.pack(f"<{num_samples}h", *([0] * num_samples))

    async def synthesize_stream(
        self, text: str, sample_rate: int = 8000
    ) -> AsyncGenerator[bytes, None]:
        """流式生成静音 PCM，分块 yield."""
        if not text or not text.strip():
            return

        duration_sec = max(len(text) * 0.3, 0.5)
        num_samples = int(sample_rate * duration_sec)
        # 每块 1024 个采样点
        chunk_samples = 1024
        offset = 0
        while offset < num_samples:
            end = min(offset + chunk_samples, num_samples)
            count = end - offset
            yield struct.pack(f"<{count}h", *([0] * count))
            offset = end
