"""音频缓冲 + VAD 单元测试."""

import asyncio
import struct

import pytest

from hxq_clink_agent.audio_buffer import AudioBuffer


def _make_pcm_frame(amplitude: int = 1000, num_samples: int = 2048) -> bytes:
    """生成指定振幅的 PCM 帧."""
    return struct.pack(f"<{num_samples}h", *([amplitude] * num_samples))


def _make_silence_frame(num_samples: int = 2048) -> bytes:
    """生成静音 PCM 帧."""
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


class TestAudioBuffer:
    """AudioBuffer VAD 测试."""

    @pytest.fixture
    def collected_segments(self):
        """收集产出的语音段."""
        return []

    @pytest.fixture
    def buffer(self, collected_segments):
        """创建测试用 AudioBuffer."""

        async def on_speech(data: bytes):
            collected_segments.append(data)

        return AudioBuffer(
            sample_rate=8000,
            sample_width=16,
            silence_sec=0.5,  # 0.5秒静音阈值
            energy_threshold=500,
            on_speech=on_speech,
        )

    @pytest.mark.asyncio
    async def test_no_speech_no_output(self, buffer, collected_segments):
        """纯静音不应产出任何语音段."""
        for _ in range(10):
            await buffer.feed(_make_silence_frame())
        assert len(collected_segments) == 0

    @pytest.mark.asyncio
    async def test_speech_then_silence_emits_segment(self, buffer, collected_segments):
        """语音后接足够长的静音应产出一段语音."""
        # 喂入语音帧
        speech_frame = _make_pcm_frame(amplitude=2000, num_samples=2048)
        for _ in range(5):
            await buffer.feed(speech_frame)

        # 喂入足够长的静音触发切割
        # 0.5s * 8000 * 2 = 8000 bytes 的静音才能触发
        silence_frame = _make_silence_frame(num_samples=2048)
        for _ in range(5):  # 5 * 4096 = 20480 bytes > 8000
            await buffer.feed(silence_frame)

        assert len(collected_segments) == 1
        assert len(collected_segments[0]) > 0

    @pytest.mark.asyncio
    async def test_multiple_speech_segments(self, buffer, collected_segments):
        """多段语音应分别产出."""
        speech = _make_pcm_frame(amplitude=2000, num_samples=2048)
        silence = _make_silence_frame(num_samples=4000)

        # 第一段
        for _ in range(3):
            await buffer.feed(speech)
        for _ in range(3):
            await buffer.feed(silence)

        # 第二段
        for _ in range(3):
            await buffer.feed(speech)
        for _ in range(3):
            await buffer.feed(silence)

        assert len(collected_segments) == 2

    @pytest.mark.asyncio
    async def test_flush_emits_remaining(self, buffer, collected_segments):
        """flush 应产出缓冲区中剩余的语音."""
        speech = _make_pcm_frame(amplitude=2000, num_samples=2048)
        for _ in range(3):
            await buffer.feed(speech)

        # 没有足够静音，不应自动产出
        assert len(collected_segments) == 0

        # flush 强制产出
        await buffer.flush()
        assert len(collected_segments) == 1

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, buffer, collected_segments):
        """reset 后应清空缓冲状态."""
        speech = _make_pcm_frame(amplitude=2000, num_samples=2048)
        for _ in range(3):
            await buffer.feed(speech)

        buffer.reset()
        await buffer.flush()
        # reset 后 flush 不应产出（因为状态已清空）
        assert len(collected_segments) == 0

    def test_compute_energy(self, buffer):
        """能量计算应返回正确值."""
        # 全零 = 能量 0
        silence = _make_silence_frame(num_samples=100)
        assert buffer._compute_energy(silence) == 0.0

        # 全 1000 = 平均能量 1000
        loud = _make_pcm_frame(amplitude=1000, num_samples=100)
        assert buffer._compute_energy(loud) == 1000.0
