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
    use_stub: bool = False  # 使用 Stub 适配器（开发调试用，向后兼容，为 True 时覆盖所有 provider 为 stub）

    # ── Provider 选择（可通过环境变量切换实现） ──
    asr_provider: str = "dashscope"  # 可选: stub | dashscope | funasr
    llm_provider: str = "openai"    # 可选: stub | openai
    tts_provider: str = "dashscope"  # 可选: stub | dashscope

    # ── DashScope 配置 ──
    dashscope_api_key: str = ""

    # ── ASR 配置 ──
    asr_model: str = "paraformer-realtime-8k-v2"
    asr_streaming_enabled: bool = True  # 是否启用流式ASR（False时回退到本地VAD+非流式）
    asr_max_sentence_silence: int = 800  # 服务端VAD静音断句阈值(ms)，范围200-6000

    # ── 打断（Barge-in）配置 ──
    barge_in_enabled: bool = True  # 是否启用语音打断（用户说话时取消当前LLM/TTS生成）

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
    log_file: str = "logs/app.log"  # 日志文件路径
    log_rotation: str = "10 MB"  # 单文件达到此大小自动轮转
    log_retention: str = "7 days"  # 日志保留时间
