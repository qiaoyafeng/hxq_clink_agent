"""适配器包 - ASR / LLM / TTS 的具体实现."""

from .asr_dashscope import ASRDashScope
from .asr_dashscope_streaming import ASRStreamingDashScope
from .asr_stub import ASRStub
from .factory import create_asr, create_asr_streaming, create_llm, create_tts
from .llm_openai import LLMOpenAI
from .llm_stub import LLMStub
from .tts_dashscope import TTSDashScope
from .tts_stub import TTSStub

__all__ = [
    "ASRStub",
    "ASRDashScope",
    "ASRStreamingDashScope",
    "LLMStub",
    "LLMOpenAI",
    "TTSStub",
    "TTSDashScope",
    "create_asr",
    "create_asr_streaming",
    "create_llm",
    "create_tts",
]
