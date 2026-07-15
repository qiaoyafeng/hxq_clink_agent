"""管线编排 - ASR → LLM → TTS 全链路处理."""

import asyncio

from loguru import logger

from .interfaces import ASRInterface, LLMInterface, TTSInterface


class Pipeline:
    """语音对话管线：ASR → LLM → TTS.

    维护每个会话的对话历史，串行处理语音段。
    """

    def __init__(
        self,
        asr: ASRInterface,
        llm: LLMInterface,
        tts: TTSInterface,
        sample_rate: int = 8000,
    ):
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._sample_rate = sample_rate
        self._history: list[dict[str, str]] = []
        self._processing = False

    @property
    def is_processing(self) -> bool:
        """当前是否正在处理语音段."""
        return self._processing

    async def process(self, pcm: bytes) -> bytes | None:
        """处理一段语音：ASR → LLM → TTS.

        Args:
            pcm: VAD 切割后的完整语音段 PCM 数据

        Returns:
            TTS 合成的 PCM 回复音频；若处理失败返回 None
        """
        self._processing = True
        try:
            # 1. ASR：语音转文本
            logger.info(f"Pipeline: ASR processing {len(pcm)} bytes")
            text = await self._asr.recognize(pcm, self._sample_rate)
            if not text or not text.strip():
                logger.debug("Pipeline: ASR returned empty text, skipping")
                return None
            logger.info(f"Pipeline: ASR result: {text}")

            # 2. LLM：生成回复
            self._history.append({"role": "user", "content": text})
            reply = await self._llm.chat(text, self._history)
            if not reply or not reply.strip():
                logger.debug("Pipeline: LLM returned empty reply, skipping")
                return None
            self._history.append({"role": "assistant", "content": reply})
            logger.info(f"Pipeline: LLM reply: {reply}")

            # 3. TTS：文本转语音
            tts_pcm = await self._tts.synthesize(reply, self._sample_rate)
            logger.info(f"Pipeline: TTS generated {len(tts_pcm)} bytes")

            return tts_pcm

        except Exception as e:
            logger.error(f"Pipeline: processing error: {e}")
            return None
        finally:
            self._processing = False

    def clear_history(self) -> None:
        """清除对话历史."""
        self._history.clear()

    def interrupt(self) -> None:
        """打断当前处理（预留接口，后续可扩展取消逻辑）."""
        logger.debug("Pipeline: interrupt requested")
