"""语音到语音流式接口 - 音频直接进、音频直接出.

与传统 ASR→LLM→TTS 三步管线不同，Voice-to-Voice 接口将完整语音对话
能力封装为单一接口：推送 PCM 音频帧，异步获取回复 PCM 音频和文本事件。
由底层实现（如百度RTC大模型互动服务）在云端完成全部 ASR+LLM+TTS。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VoiceEvent:
    """语音对话事件.

    Attributes:
        event_type: 事件类型，可选值:
            - "media_ready": 媒体通道就绪
            - "asr_text": 用户语音识别文本
            - "llm_reply": AI 回复文本
            - "tts_begin": TTS 开始播报
            - "tts_end": TTS 结束播报
            - "interrupt_word": 命中打断词
            - "function_call": 函数调用
            - "error": 错误事件
        content: 事件内容文本
    """

    event_type: str
    content: str = ""


class VoiceToVoiceInterface(ABC):
    """语音到语音流式抽象基类.

    定义实时语音对话的统一接口：持续推送 PCM 音频帧，
    异步获取回复 PCM 音频和文本事件（ASR结果/LLM回复/TTS状态等）。
    支持打断（interrupt）场景。
    """

    @abstractmethod
    async def start(self) -> None:
        """启动语音对话会话（建立到云端服务的连接）."""
        ...

    @abstractmethod
    def feed(self, pcm: bytes) -> None:
        """推送上行 PCM 音频帧.

        Args:
            pcm: 原始 PCM 字节（16bit signed LE），采样率由实现方决定
        """
        ...

    @abstractmethod
    async def get_audio_chunk(self) -> bytes | None:
        """异步获取下行 PCM 音频块.

        Returns:
            PCM 字节数据块（16bit signed LE, 8kHz）；若会话结束返回 None
        """
        ...

    @abstractmethod
    async def get_event(self) -> VoiceEvent | None:
        """异步获取文本事件.

        Returns:
            VoiceEvent 实例；若会话结束返回 None
        """
        ...

    @abstractmethod
    async def interrupt(self) -> None:
        """打断当前 TTS 播报."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """停止会话并释放资源."""
        ...
