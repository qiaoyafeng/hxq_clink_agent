"""入口模块 - 启动统一的 FastAPI 服务."""

import sys

import uvicorn
from loguru import logger

from .config import Settings
from .health import set_session_count_getter
from .ws_server import get_active_session_count


def main() -> None:
    """应用入口：加载配置并启动 FastAPI 服务."""
    settings = Settings()

    # 配置 loguru
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # 注入会话数获取函数到健康检查模块
    set_session_count_getter(get_active_session_count)

    logger.info("hxq_clink_agent starting...")
    logger.info(f"  Server    : {settings.host}:{settings.port}")
    logger.info(f"  WS path   : {settings.ws_path}")
    logger.info(f"  PCM: {settings.pcm_sample_rate}Hz / {settings.pcm_sample_width}bit")
    logger.info(f"  VAD: silence={settings.vad_silence_sec}s, threshold={settings.vad_energy_threshold}")
    if settings.use_stub:
        logger.info("  Adapters  : STUB (ASRStub / LLMStub / TTSStub)")
    else:
        logger.info(f"  ASR       : {settings.asr_model}")
        logger.info(f"  LLM       : {settings.llm_model} @ {settings.llm_base_url}")
        logger.info(f"  TTS       : {settings.tts_model} (voice={settings.tts_voice})")

    uvicorn.run(
        "hxq_clink_agent.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=settings.access_log,
    )


if __name__ == "__main__":
    main()
