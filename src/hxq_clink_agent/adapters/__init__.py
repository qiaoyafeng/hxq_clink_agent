"""适配器包 - ASR / LLM / TTS 的具体实现."""

from .asr_stub import ASRStub
from .llm_stub import LLMStub
from .tts_stub import TTSStub

__all__ = ["ASRStub", "LLMStub", "TTSStub"]
