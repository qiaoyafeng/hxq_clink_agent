"""管线编排 - ASR → LLM → TTS 全链路处理."""

import asyncio
import time

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
        total_start = time.monotonic()
        try:
            # 1. ASR：语音转文本
            logger.info(f"Pipeline: ASR processing {len(pcm)} bytes")
            t0 = time.monotonic()
            text = await self._asr.recognize(pcm, self._sample_rate)
            asr_elapsed = time.monotonic() - t0
            if not text or not text.strip():
                logger.debug(f"Pipeline: ASR returned empty text ({asr_elapsed:.2f}s), skipping")
                return None
            logger.info(f"Pipeline: ASR done in {asr_elapsed:.2f}s | text: {text!r}")

            # 2. LLM：生成回复
            self._history.append({"role": "user", "content": text})
            t0 = time.monotonic()
            reply = await self._llm.chat(text, self._history)
            llm_elapsed = time.monotonic() - t0
            if not reply or not reply.strip():
                logger.debug(f"Pipeline: LLM returned empty reply ({llm_elapsed:.2f}s), skipping")
                return None
            self._history.append({"role": "assistant", "content": reply})
            logger.info(f"Pipeline: LLM done in {llm_elapsed:.2f}s | reply: {reply!r}")

            # 3. TTS：文本转语音
            t0 = time.monotonic()
            tts_pcm = await self._tts.synthesize(reply, self._sample_rate)
            tts_elapsed = time.monotonic() - t0
            logger.info(f"Pipeline: TTS done in {tts_elapsed:.2f}s | {len(tts_pcm)} bytes")

            total_elapsed = time.monotonic() - total_start
            logger.info(f"Pipeline: total {total_elapsed:.2f}s")

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
