"""抽象接口包 - ASR / LLM / TTS 的基类定义."""

from .asr import ASRInterface
from .llm import LLMInterface
from .tts import TTSInterface

__all__ = ["ASRInterface", "LLMInterface", "TTSInterface"]
