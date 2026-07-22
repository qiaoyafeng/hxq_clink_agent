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

    # ── 并发控制 ──
    max_concurrent_sessions: int = 10  # 同时进行的最大会话数，0 表示不限制

    # ── 鉴权配置（用于验证天润融通客户端签名） ──
    auth_enabled: bool = True  # 是否开启 WebSocket 签名验证
    app_id: str = ""
    access_key_id: str = ""
    access_key_secret: str = ""

    # ── PCM 音频参数 ──
    pcm_sample_rate: int = 8000
    pcm_sample_width: int = 16  # bits
    pcm_frame_size: int = 960  # bytes per frame（协议要求 60ms/帧，8k*16bit*60ms=960B）
    pcm_frame_interval: float = 0.06  # seconds between frames（60ms）

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

    # ── 转人工（Transfer to Agent）配置 ──
    transfer_enabled: bool = True  # 是否启用转人工功能（关键词识别触发）
    transfer_keywords: str = "转人工,人工客服,人工服务,转接人工,找人工"  # 触发关键词（逗号分隔）
    transfer_qno: str = "9999"  # 转人工目标队列号

    # ── LLM 配置（OpenAI 兼容接口） ──
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen-turbo"
    llm_system_prompt: str = "你是一个智能语音助手，请用简洁的语言回答用户的问题。"

    # ── TTS 配置（DashScope REST API） ──
    tts_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    tts_model: str = "cosyvoice-v2-0.5b"
    tts_voice: str = "longxiaochun_v2"
    tts_sample_rate: int = 22050

    # ── Voice-to-Voice Provider（空值=使用ASR-LLM-TTS管线；"baidu"=使用百度RTC） ──
    voice_provider: str = ""  # 可选: baidu（留空则使用 ASR-LLM-TTS 管线）

    # ── 百度RTC大模型互动服务配置 ──
    baidu_rtc_app_id: str = ""  # 百度RTC互动应用ID
    baidu_rtc_ak: str = ""  # 百度云 Access Key ID
    baidu_rtc_sk: str = ""  # 百度云 Secret Access Key
    baidu_rtc_license_key: str = ""  # 设备 License Key
    baidu_rtc_device_id: str = "hxq_clink_agent"  # 设备唯一标识
    baidu_rtc_user_id: str = "default_user"  # 用户唯一标识
    baidu_rtc_api_endpoint: str = "rtc-aiagent.baidubce.com"  # REST API 域名
    baidu_rtc_ws_endpoint: str = "wss://rtc-aiotgw.exp.bcelive.com/v1/realtime"  # WebSocket 端侧接入地址
    baidu_rtc_audio_codec: str = "raw16k"  # 音频编码: raw16k | pcmu | opus_cbr_16000

    # 百度RTC模式选择
    baidu_rtc_e2e_enabled: bool = False  # True=端到端语音模型(audio_to_audio), False=传统托管模式(ASR-LLM-TTS)
    baidu_rtc_e2e_prompt: str = ""  # 端到端模型角色提示词
    baidu_rtc_e2e_vcn: int = 8003  # 端到端模型音色(8003/8014/8008/8021)

    # 百度RTC传统托管模式配置
    baidu_rtc_scene_role_name: str = ""  # 控制台预设角色名（与 scene_role_prompt 二选一）
    baidu_rtc_scene_role_prompt: str = ""  # 临时角色 prompt
    baidu_rtc_tts_vcn: str = ""  # TTS 发音人
    baidu_rtc_tts_sayhi: str = ""  # 招呼语
    baidu_rtc_lang: str = "zh"  # 语言
    baidu_rtc_disable_auto_interrupt: bool = False  # 是否关闭云端自动打断
    baidu_rtc_asr_vad: int = 200  # ASR 断句等待时长(ms)

    # ── 日志 ──
    log_level: str = "INFO"
    access_log: bool = False
    log_file: str = "logs/app_{time:YYYY-MM-DD}.log"  # 日志文件路径（支持 {time} 占位符，按日期生成）
    log_rotation: str = "00:00"  # 每天午夜轮转
    log_retention: str = "7 days"  # 日志保留时间
