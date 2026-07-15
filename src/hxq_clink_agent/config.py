"""应用配置 - 使用 pydantic-settings 从环境变量/.env 加载."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """天润融通 PCM 语音流 FastAPI Server 配置."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="HXQ_",
        extra="ignore",
    )

    # ── 服务监听配置 ──
    host: str = "0.0.0.0"
    port: int = 8000
    ws_path: str = "/realtime_voice"

    # ── 鉴权配置（用于验证天润融通客户端签名） ──
    auth_enabled: bool = True  # 是否开启 WebSocket 签名验证
    app_id: str = ""
    access_key_id: str = ""
    access_key_secret: str = ""

    # ── PCM 音频参数 ──
    pcm_sample_rate: int = 8000
    pcm_sample_width: int = 16  # bits
    pcm_frame_size: int = 4096  # bytes per frame
    pcm_frame_interval: float = 0.25  # seconds between frames

    # ── VAD 参数 ──
    vad_silence_sec: float = 0.8
    vad_energy_threshold: int = 500

    # ── 运行模式 ──
    production: bool = True  # 生产环境关闭 Swagger UI 等调试工具
    use_stub: bool = False  # 使用 Stub 适配器（开发调试用）

    # ── DashScope 配置 ──
    dashscope_api_key: str = ""

    # ── ASR 配置 ──
    asr_model: str = "paraformer-realtime-8k-v2"

    # ── LLM 配置（OpenAI 兼容接口） ──
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen-turbo"
    llm_system_prompt: str = "你是一个智能语音助手，请用简洁的语言回答用户的问题。"

    # ── TTS 配置（DashScope REST API） ──
    tts_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    tts_model: str = "cosyvoice-v2-0.5b"
    tts_voice: str = "longxiaochun_v2"
    tts_sample_rate: int = 22050

    # ── 日志 ──
    log_level: str = "INFO"
    access_log: bool = False
