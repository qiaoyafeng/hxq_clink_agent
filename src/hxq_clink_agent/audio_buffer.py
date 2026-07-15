"""音频缓冲 + 简易 VAD - 将连续 PCM 流切割为语音段."""

import asyncio
import struct
from collections.abc import Callable, Coroutine
from typing import Any

from loguru import logger


class AudioBuffer:
    """PCM 音频缓冲区，内置简易能量 VAD.

    当检测到语音段结束（静音超过阈值）时，通过回调产出完整语音段。
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        sample_width: int = 16,
        silence_sec: float = 0.8,
        energy_threshold: int = 500,
        on_speech: Callable[[bytes], Coroutine[Any, Any, None]] | None = None,
    ):
        self._sample_rate = sample_rate
        self._bytes_per_sample = sample_width // 8
        self._silence_threshold_bytes = int(
            silence_sec * sample_rate * self._bytes_per_sample
        )
        self._energy_threshold = energy_threshold
        self._on_speech = on_speech

        # 内部状态
        self._buffer = bytearray()
        self._speech_started = False
        self._silence_bytes = 0

    def set_callback(self, on_speech: Callable[[bytes], Coroutine[Any, Any, None]]) -> None:
        """设置语音段产出回调."""
        self._on_speech = on_speech

    async def feed(self, data: bytes) -> None:
        """喂入 PCM 数据，内部进行 VAD 检测.

        Args:
            data: 原始 PCM 字节（16bit signed LE）
        """
        frame_energy = self._compute_energy(data)

        if frame_energy >= self._energy_threshold:
            # 有语音活动
            if not self._speech_started:
                self._speech_started = True
                logger.debug("VAD: speech started")
            self._buffer.extend(data)
            self._silence_bytes = 0
        else:
            # 静音
            if self._speech_started:
                self._buffer.extend(data)
                self._silence_bytes += len(data)

                if self._silence_bytes >= self._silence_threshold_bytes:
                    # 静音足够长，切割语音段
                    await self._emit_speech()

    async def flush(self) -> None:
        """强制产出缓冲区中剩余的语音数据（用于连接结束时）."""
        if self._speech_started and len(self._buffer) > 0:
            await self._emit_speech()

    async def _emit_speech(self) -> None:
        """产出语音段并重置状态."""
        # 去掉尾部静音部分
        speech_end = len(self._buffer) - self._silence_bytes
        speech_data = bytes(self._buffer[:speech_end])

        # 重置
        self._buffer.clear()
        self._speech_started = False
        self._silence_bytes = 0

        if speech_data and self._on_speech:
            logger.debug(f"VAD: speech segment emitted, {len(speech_data)} bytes")
            await self._on_speech(speech_data)

    def _compute_energy(self, data: bytes) -> float:
        """计算 PCM 帧的平均绝对能量."""
        if len(data) < self._bytes_per_sample:
            return 0.0

        num_samples = len(data) // self._bytes_per_sample
        # 解码 16bit signed LE samples
        samples = struct.unpack(f"<{num_samples}h", data[: num_samples * 2])
        if not samples:
            return 0.0
        return sum(abs(s) for s in samples) / len(samples)

    def reset(self) -> None:
        """重置缓冲区状态."""
        self._buffer.clear()
        self._speech_started = False
        self._silence_bytes = 0
