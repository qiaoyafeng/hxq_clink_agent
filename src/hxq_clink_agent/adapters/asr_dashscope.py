"""ASR 适配器 - 通过 DashScope SDK 调用阿里云 Paraformer 语音识别."""

import os
import tempfile
import wave
from http import HTTPStatus

import dashscope
from dashscope.audio.asr import Recognition
from loguru import logger

from ..interfaces.asr import ASRInterface


class ASRDashScope(ASRInterface):
    """阿里云 Paraformer ASR 适配器（DashScope SDK）.

    使用 paraformer-realtime 系列模型，将 PCM 音频通过 WebSocket
    发送至阿里云进行语音识别。

    Args:
        api_key: DashScope API Key
        model: ASR 模型名称，默认 paraformer-realtime-8k-v2（8kHz 电话场景）
    """

    def __init__(self, api_key: str, model: str = "paraformer-realtime-8k-v2"):
        self._api_key = api_key
        self._model = model
        # 设置 DashScope SDK 全局 API Key
        dashscope.api_key = api_key

    async def recognize(self, pcm: bytes, sample_rate: int = 8000) -> str:
        """将 PCM 音频数据转为文本.

        实现步骤：
        1. 将 raw PCM bytes 封装为 WAV 格式（in-memory + 临时文件）
        2. 调用 DashScope Recognition.call() 进行非流式识别
        3. 拼接识别结果中的句子文本并返回
        """
        if not pcm:
            return ""

        wav_path = None
        try:
            # 将 raw PCM 写入临时 WAV 文件
            wav_path = self._pcm_to_wav(pcm, sample_rate)

            # 调用 DashScope Recognition（SDK 内部使用 WebSocket）
            recognition = Recognition(
                model=self._model,
                format="wav",
                sample_rate=sample_rate,
                callback=None,
            )
            # call() 是同步阻塞的，对于几秒的 VAD 语音段可接受
            result = recognition.call(wav_path)

            if result.status_code != HTTPStatus.OK:
                logger.error(
                    f"ASR DashScope error: status={result.status_code}, "
                    f"message={getattr(result, 'message', 'unknown')}"
                )
                return ""

            sentences = result.get_sentence()
            if not sentences:
                return ""

            # 拼接所有识别出的句子
            text = "".join(s.get("text", "") for s in sentences).strip()
            logger.debug(f"ASR result: {text!r}")
            return text

        except Exception as e:
            logger.error(f"ASR DashScope exception: {e}")
            return ""
        finally:
            # 清理临时文件
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def _pcm_to_wav(self, pcm: bytes, sample_rate: int) -> str:
        """将 raw PCM (16bit signed LE, mono) 写入临时 WAV 文件并返回路径."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)          # mono
            wf.setsampwidth(2)          # 16bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return tmp.name
