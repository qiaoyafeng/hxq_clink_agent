"""TTS 占位适配器 - 开发/测试用."""

import struct

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
