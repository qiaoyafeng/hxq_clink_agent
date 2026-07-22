"""抽象接口包 - ASR / LLM / TTS / Voice-to-Voice 的基类定义."""

from .asr import ASRInterface
from .asr_streaming import ASRStreamingInterface
from .llm import LLMInterface
from .tts import TTSInterface
from .voice_to_voice import VoiceEvent, VoiceToVoiceInterface

__all__ = [
    "ASRInterface",
    "ASRStreamingInterface",
    "LLMInterface",
    "TTSInterface",
    "VoiceToVoiceInterface",
    "VoiceEvent",
]
