"""管线编排单元测试."""

import pytest

from hxq_clink_agent.adapters import ASRStub, LLMStub, TTSStub
from hxq_clink_agent.pipeline import Pipeline


@pytest.fixture
def pipeline():
    """创建使用 Stub 适配器的测试管线."""
    return Pipeline(
        asr=ASRStub(),
        llm=LLMStub(),
        tts=TTSStub(),
        sample_rate=8000,
    )


class TestPipeline:
    """Pipeline 管线测试."""

    @pytest.mark.asyncio
    async def test_process_returns_pcm(self, pipeline):
        """正常处理应返回 TTS PCM 数据."""
        # 模拟 PCM 输入（随便什么字节都行，Stub ASR 不解析）
        fake_pcm = b"\x00\x01" * 4096
        result = await pipeline.process(fake_pcm)

        assert result is not None
        assert len(result) > 0
        # TTS Stub 返回 16bit PCM，长度应为偶数
        assert len(result) % 2 == 0

    @pytest.mark.asyncio
    async def test_history_accumulates(self, pipeline):
        """对话历史应在每次处理后累积."""
        fake_pcm = b"\x00\x01" * 100

        await pipeline.process(fake_pcm)
        assert len(pipeline._history) == 2  # user + assistant

        await pipeline.process(fake_pcm)
        assert len(pipeline._history) == 4  # 2 rounds

    @pytest.mark.asyncio
    async def test_clear_history(self, pipeline):
        """clear_history 应清空对话历史."""
        fake_pcm = b"\x00\x01" * 100
        await pipeline.process(fake_pcm)
        assert len(pipeline._history) > 0

        pipeline.clear_history()
        assert len(pipeline._history) == 0

    @pytest.mark.asyncio
    async def test_is_processing_flag(self, pipeline):
        """处理期间 is_processing 应为 True."""
        assert pipeline.is_processing is False

        fake_pcm = b"\x00\x01" * 100
        await pipeline.process(fake_pcm)

        # 处理完成后应为 False
        assert pipeline.is_processing is False
