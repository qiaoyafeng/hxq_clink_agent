"""适配器工厂 - 根据配置动态实例化 ASR / LLM / TTS 适配器.

通过注册表模式管理所有可用的 Provider 实现。
扩展新 Provider 只需：
1. 实现对应接口类
2. 在注册表 dict 中增加一行映射
"""

from __future__ import annotations

from loguru import logger

from ..config import Settings
from ..interfaces.asr import ASRInterface
from ..interfaces.asr_streaming import ASRStreamingInterface
from ..interfaces.llm import LLMInterface
from ..interfaces.tts import TTSInterface
from .asr_dashscope import ASRDashScope
from .asr_dashscope_streaming import ASRStreamingDashScope
from .asr_stub import ASRStub
from .llm_openai import LLMOpenAI
from .llm_stub import LLMStub
from .tts_dashscope import TTSDashScope
from .tts_stub import TTSStub

# ── ASR 注册表 ──
_ASR_REGISTRY: dict[str, type[ASRInterface]] = {
    "stub": ASRStub,
    "dashscope": ASRDashScope,
}

# ── 流式 ASR 注册表 ──
_ASR_STREAMING_REGISTRY: dict[str, type[ASRStreamingInterface]] = {
    "dashscope": ASRStreamingDashScope,
}

# ── LLM 注册表 ──
_LLM_REGISTRY: dict[str, type[LLMInterface]] = {
    "stub": LLMStub,
    "openai": LLMOpenAI,
}

# ── TTS 注册表 ──
_TTS_REGISTRY: dict[str, type[TTSInterface]] = {
    "stub": TTSStub,
    "dashscope": TTSDashScope,
}


def _resolve_provider(settings: Settings) -> tuple[str, str, str]:
    """解析最终的 provider 名称（兼容 use_stub）."""
    asr = settings.asr_provider
    llm = settings.llm_provider
    tts = settings.tts_provider

    # 向后兼容：use_stub=True 覆盖所有 provider
    if settings.use_stub:
        asr = llm = tts = "stub"

    return asr, llm, tts


def create_asr(settings: Settings) -> ASRInterface:
    """根据配置创建 ASR 适配器实例.

    Raises:
        ValueError: 未知的 asr_provider
    """
    provider, _, _ = _resolve_provider(settings)

    if provider not in _ASR_REGISTRY:
        available = ", ".join(sorted(_ASR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown asr_provider: {provider!r}. Available: {available}"
        )

    cls = _ASR_REGISTRY[provider]

    # 根据不同 Provider 传入对应参数
    if provider == "stub":
        instance = cls()
    elif provider == "dashscope":
        instance = cls(
            api_key=settings.dashscope_api_key,
            model=settings.asr_model,
        )
    else:
        # 通用兜底：尝试无参构造（扩展 Provider 可能需要自定义逻辑）
        instance = cls()

    logger.info(f"ASR adapter created: provider={provider}, class={cls.__name__}")
    return instance


def create_asr_streaming(settings: Settings) -> ASRStreamingInterface | None:
    """根据配置创建流式 ASR 适配器实例.

    Returns:
        流式 ASR 实例；若 Provider 不支持流式或未启用则返回 None
    """
    provider, _, _ = _resolve_provider(settings)

    # 未启用流式 ASR 或 Provider 不在流式注册表中
    if not settings.asr_streaming_enabled:
        logger.info("ASR streaming disabled by config")
        return None

    if provider not in _ASR_STREAMING_REGISTRY:
        logger.info(
            f"ASR streaming not available for provider={provider!r}, "
            f"falling back to local VAD mode"
        )
        return None

    cls = _ASR_STREAMING_REGISTRY[provider]

    # 根据不同 Provider 传入对应参数
    if provider == "dashscope":
        instance = cls(
            api_key=settings.dashscope_api_key,
            model=settings.asr_model,
            sample_rate=settings.pcm_sample_rate,
            max_sentence_silence=settings.asr_max_sentence_silence,
        )
    else:
        instance = cls()

    logger.info(
        f"ASR streaming adapter created: provider={provider}, class={cls.__name__}"
    )
    return instance


def create_llm(settings: Settings) -> LLMInterface:
    """根据配置创建 LLM 适配器实例.

    Raises:
        ValueError: 未知的 llm_provider
    """
    _, provider, _ = _resolve_provider(settings)

    if provider not in _LLM_REGISTRY:
        available = ", ".join(sorted(_LLM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown llm_provider: {provider!r}. Available: {available}"
        )

    cls = _LLM_REGISTRY[provider]

    if provider == "stub":
        instance = cls()
    elif provider == "openai":
        instance = cls(
            api_key=settings.dashscope_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            system_prompt=settings.llm_system_prompt,
        )
    else:
        instance = cls()

    logger.info(f"LLM adapter created: provider={provider}, class={cls.__name__}")
    return instance


def create_tts(settings: Settings) -> TTSInterface:
    """根据配置创建 TTS 适配器实例.

    Raises:
        ValueError: 未知的 tts_provider
    """
    _, _, provider = _resolve_provider(settings)

    if provider not in _TTS_REGISTRY:
        available = ", ".join(sorted(_TTS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown tts_provider: {provider!r}. Available: {available}"
        )

    cls = _TTS_REGISTRY[provider]

    if provider == "stub":
        instance = cls()
    elif provider == "dashscope":
        instance = cls(
            api_key=settings.dashscope_api_key,
            base_url=settings.tts_base_url,
            model=settings.tts_model,
            voice=settings.tts_voice,
        )
    else:
        instance = cls()

    logger.info(f"TTS adapter created: provider={provider}, class={cls.__name__}")
    return instance
