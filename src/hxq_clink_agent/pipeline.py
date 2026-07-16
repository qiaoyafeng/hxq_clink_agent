"""管线编排 - ASR → LLM → TTS 全链路处理."""

import asyncio
import time
from collections.abc import AsyncGenerator

from loguru import logger

from .interfaces import ASRInterface, LLMInterface, TTSInterface

# 断句符号：遇到这些字符时立即切分
_SENTENCE_DELIMITERS = set("。！？；\n")
# 弱断句符：超过最大长度时在这些字符处切分
_WEAK_DELIMITERS = set("，、,")
# 缓冲区最大字符数
_MAX_BUFFER_LEN = 50


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

    async def process_text(self, text: str) -> bytes | None:
        """从文本开始处理：LLM → TTS（跳过 ASR）.

        用于流式 ASR 已识别出完整句子后的后续处理。

        Args:
            text: 已识别的用户语音文本

        Returns:
            TTS 合成的 PCM 回复音频；若处理失败返回 None
        """
        if not text or not text.strip():
            return None

        self._processing = True
        total_start = time.monotonic()
        try:
            logger.info(f"Pipeline: process_text | input: {text!r}")

            # 1. LLM：生成回复
            self._history.append({"role": "user", "content": text})
            t0 = time.monotonic()
            reply = await self._llm.chat(text, self._history)
            llm_elapsed = time.monotonic() - t0
            if not reply or not reply.strip():
                logger.debug(f"Pipeline: LLM returned empty reply ({llm_elapsed:.2f}s), skipping")
                return None
            self._history.append({"role": "assistant", "content": reply})
            logger.info(f"Pipeline: LLM done in {llm_elapsed:.2f}s | reply: {reply!r}")

            # 2. TTS：文本转语音
            t0 = time.monotonic()
            tts_pcm = await self._tts.synthesize(reply, self._sample_rate)
            tts_elapsed = time.monotonic() - t0
            logger.info(f"Pipeline: TTS done in {tts_elapsed:.2f}s | {len(tts_pcm)} bytes")

            total_elapsed = time.monotonic() - total_start
            logger.info(f"Pipeline: process_text total {total_elapsed:.2f}s")

            return tts_pcm

        except Exception as e:
            logger.error(f"Pipeline: process_text error: {e}")
            return None
        finally:
            self._processing = False

    def clear_history(self) -> None:
        """清除对话历史."""
        self._history.clear()

    def interrupt(self) -> None:
        """打断当前处理（预留接口，后续可扩展取消逻辑）."""
        logger.debug("Pipeline: interrupt requested")

    # ── 流式处理方法 ──

    async def process_text_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """流式处理文本：LLM 流式输出 → 断句 → TTS 流式合成，逐块 yield PCM.

        LLM 边生成边按句子切分，每个句子立即送入 TTS 流式合成，
        TTS 每生成一个 PCM 块就立即 yield，大幅降低首字节延迟。

        Args:
            text: 已识别的用户语音文本

        Yields:
            TTS 合成的 PCM 字节数据块
        """
        if not text or not text.strip():
            return

        self._processing = True
        total_start = time.monotonic()
        try:
            logger.info(f"Pipeline: process_text_stream | input: {text!r}")

            # 将用户输入加入历史
            self._history.append({"role": "user", "content": text})

            # LLM 流式生成 + 断句 + TTS 流式合成
            text_buffer = ""
            full_reply = ""
            t_llm_start = time.monotonic()
            first_token = True

            async for token in self._llm.chat_stream(text, self._history):
                if first_token:
                    logger.info(
                        f"Pipeline: LLM first token in "
                        f"{time.monotonic() - t_llm_start:.2f}s"
                    )
                    first_token = False

                full_reply += token
                text_buffer += token

                # 尝试断句
                segments = self._segment_buffer(text_buffer, flush=False)
                for segment in segments[:-1]:
                    # 切出的完整句子立即送入 TTS
                    async for pcm_chunk in self._tts.synthesize_stream(
                        segment, self._sample_rate
                    ):
                        yield pcm_chunk
                # 最后一个是未完成的缓冲区残余
                text_buffer = segments[-1] if segments else text_buffer

            # LLM 流结束，刷出缓冲区剩余文本
            llm_elapsed = time.monotonic() - t_llm_start
            logger.info(f"Pipeline: LLM stream done in {llm_elapsed:.2f}s | reply: {full_reply!r}")

            if text_buffer.strip():
                async for pcm_chunk in self._tts.synthesize_stream(
                    text_buffer, self._sample_rate
                ):
                    yield pcm_chunk

            # 将完整回复加入历史
            if full_reply.strip():
                self._history.append({"role": "assistant", "content": full_reply})

            total_elapsed = time.monotonic() - total_start
            logger.info(f"Pipeline: process_text_stream total {total_elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Pipeline: process_text_stream error: {e}")
        finally:
            self._processing = False

    async def process_stream(self, pcm: bytes) -> AsyncGenerator[bytes, None]:
        """流式处理语音：ASR → LLM 流式 → TTS 流式，逐块 yield PCM.

        Args:
            pcm: VAD 切割后的完整语音段 PCM 数据

        Yields:
            TTS 合成的 PCM 字节数据块
        """
        self._processing = True
        try:
            # 1. ASR：语音转文本
            logger.info(f"Pipeline: ASR processing {len(pcm)} bytes")
            t0 = time.monotonic()
            text = await self._asr.recognize(pcm, self._sample_rate)
            asr_elapsed = time.monotonic() - t0
            if not text or not text.strip():
                logger.debug(
                    f"Pipeline: ASR returned empty text ({asr_elapsed:.2f}s), skipping"
                )
                return
            logger.info(f"Pipeline: ASR done in {asr_elapsed:.2f}s | text: {text!r}")

            # 2. LLM 流式 + TTS 流式
            async for pcm_chunk in self.process_text_stream(text):
                yield pcm_chunk

        except Exception as e:
            logger.error(f"Pipeline: process_stream error: {e}")
        finally:
            self._processing = False

    def _segment_buffer(self, buffer: str, flush: bool = False) -> list[str]:
        """将文本缓冲区按句子切分.

        Args:
            buffer: 当前缯冲的文本
            flush: 是否强制刷出所有内容

        Returns:
            切分后的句子列表，最后一个元素为未完成的缓冲区剩余
        """
        if flush:
            return [buffer] if buffer else []

        segments: list[str] = []
        current = ""

        for char in buffer:
            current += char
            if char in _SENTENCE_DELIMITERS:
                # 强断句符：立即切分
                if current.strip():
                    segments.append(current)
                current = ""
            elif len(current) >= _MAX_BUFFER_LEN:
                # 超过最大长度，尝试在弱断句符处切分
                last_weak = -1
                for i in range(len(current) - 1, -1, -1):
                    if current[i] in _WEAK_DELIMITERS:
                        last_weak = i
                        break
                if last_weak > 0:
                    # 在最近的弱断句符处切分（含该符号）
                    seg = current[: last_weak + 1]
                    if seg.strip():
                        segments.append(seg)
                    current = current[last_weak + 1 :]
                else:
                    # 无弱断句符，强制切分
                    if current.strip():
                        segments.append(current)
                    current = ""

        # 最后残余作为未完成的缓冲
        segments.append(current)
        return segments
