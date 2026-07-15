"""ASR 占位适配器 - 开发/测试用."""

from ..interfaces.asr import ASRInterface


class ASRStub(ASRInterface):
    """ASR 占位实现，返回固定文本，用于开发联调."""

    async def recognize(self, pcm: bytes, sample_rate: int = 8000) -> str:
        """返回包含字节数的占位文本."""
        return f"[ASR] received {len(pcm)} bytes of audio"
