"""TTS 适配器 - 通过 DashScope REST API 调用阿里云 CosyVoice 语音合成."""

import base64
import json

import httpx
from loguru import logger

from ..interfaces.tts import TTSInterface


class TTSDashScope(TTSInterface):
    """阿里云 CosyVoice TTS 适配器（DashScope REST API + SSE 流式）.

    使用 CosyVoice 系列模型，通过 HTTP SSE 流式返回 base64 编码的
    PCM 音频数据，避免等待完整音频生成。

    Args:
        api_key: DashScope API Key
        base_url: DashScope HTTP API 基础地址
        model: TTS 模型名称，如 cosyvoice-v2-0.5b
        voice: 音色名称，如 longxiaochun_v2
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/api/v1",
        model: str = "cosyvoice-v2-0.5b",
        voice: str = "longxiaochun_v2",
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._voice = voice

    async def synthesize(self, text: str, sample_rate: int = 8000) -> bytes:
        """将文本合成为 PCM 音频数据.

        通过 SSE 流式接收 base64 编码的 PCM 数据块，
        解码后拼接为完整的 PCM 字节流返回。

        Args:
            text: 待合成的文本
            sample_rate: 目标采样率，默认 8000Hz

        Returns:
            PCM 字节数据（16bit signed LE）
        """
        if not text or not text.strip():
            return b""

        pcm_chunks: list[bytes] = []
        url = f"{self._base_url}/services/audio/tts/SpeechSynthesizer"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "X-DashScope-SSE": "enable",
                    },
                    json={
                        "model": self._model,
                        "input": {
                            "text": text,
                            "voice": self._voice,
                            "format": "pcm",
                            "sample_rate": sample_rate,
                        },
                    },
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        logger.error(
                            f"TTS DashScope error: status={response.status_code}, "
                            f"body={body.decode('utf-8', errors='replace')[:200]}"
                        )
                        return b""

                    # 解析 SSE 事件流
                    async for event_data in self._iter_sse_events(response):
                        chunk = self._extract_audio(event_data)
                        if chunk:
                            pcm_chunks.append(chunk)

            pcm = b"".join(pcm_chunks)
            logger.debug(f"TTS generated {len(pcm)} bytes from {len(text)} chars")
            return pcm

        except Exception as e:
            logger.error(f"TTS DashScope exception: {e}")
            return b""

    async def _iter_sse_events(self, response: httpx.Response):
        """解析 SSE 事件流，yield 每个事件的 data 字段内容."""
        buffer = ""
        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line:
                # 空行表示一个 SSE 事件结束
                if buffer:
                    yield buffer
                    buffer = ""
                continue
            if line.startswith("data:"):
                buffer = line[5:].strip()
            # 忽略 id:, event:, 等其他字段

        # 处理最后一个事件
        if buffer:
            yield buffer

    def _extract_audio(self, event_data: str) -> bytes | None:
        """从 SSE 事件 JSON 中提取 base64 编码的 PCM 数据."""
        try:
            data = json.loads(event_data)
            # DashScope TTS SSE 格式: {"output": {"audio": {"data": "base64..."}}}
            audio_data = data.get("output", {}).get("audio", {}).get("data")
            if audio_data:
                return base64.b64decode(audio_data)
            return None
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"TTS SSE parse skip: {e}")
            return None
