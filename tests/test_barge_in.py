"""打断（Barge-in）功能单元测试."""

import asyncio

import pytest

from hxq_clink_agent.adapters import ASRStub, LLMStub, TTSStub
from hxq_clink_agent.adapters.asr_dashscope_streaming import ASRStreamingDashScope
from hxq_clink_agent.pipeline import Pipeline


class TestASRVoiceActivityEvent:
    """ASR 流式适配器 voice activity event 测试."""

    def test_event_initial_state_is_unset(self):
        """初始状态 voice activity event 应未设置."""
        asr = ASRStreamingDashScope(api_key="fake", model="fake")
        assert not asr._voice_activity_event.is_set()

    def test_clear_voice_activity(self):
        """clear_voice_activity 应清除 event."""
        asr = ASRStreamingDashScope(api_key="fake", model="fake")
        asr._voice_activity_event.set()
        assert asr._voice_activity_event.is_set()
        asr.clear_voice_activity()
        assert not asr._voice_activity_event.is_set()

    @pytest.mark.asyncio
    async def test_wait_voice_activity_returns_on_set(self):
        """wait_voice_activity 应在 event 被 set 后返回并自动 clear."""
        asr = ASRStreamingDashScope(api_key="fake", model="fake")

        # 在短延迟后 set event
        async def set_after_delay():
            await asyncio.sleep(0.05)
            asr._voice_activity_event.set()

        asyncio.create_task(set_after_delay())
        # 应在 event set 后返回
        await asyncio.wait_for(asr.wait_voice_activity(), timeout=1.0)
        # 返回后 event 应已 clear
        assert not asr._voice_activity_event.is_set()

    @pytest.mark.asyncio
    async def test_wait_voice_activity_blocks_when_unset(self):
        """wait_voice_activity 应在 event 未 set 时阻塞."""
        asr = ASRStreamingDashScope(api_key="fake", model="fake")

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asr.wait_voice_activity(), timeout=0.1)


class TestPipelineBargeIn:
    """Pipeline barge-in 相关测试."""

    @pytest.fixture
    def pipeline(self):
        """创建使用 Stub 适配器的测试管线."""
        return Pipeline(
            asr=ASRStub(),
            llm=LLMStub(),
            tts=TTSStub(),
            sample_rate=8000,
        )

    @pytest.mark.asyncio
    async def test_pop_last_user_message(self, pipeline):
        """pop_last_user_message 应移除最后一条 user 消息."""
        pipeline._history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
        pipeline.pop_last_user_message()
        assert len(pipeline._history) == 2
        assert pipeline._history[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_pop_last_user_message_empty_history(self, pipeline):
        """空历史时 pop_last_user_message 不报错."""
        pipeline._history = []
        pipeline.pop_last_user_message()  # 不应抛出异常
        assert len(pipeline._history) == 0

    @pytest.mark.asyncio
    async def test_process_text_stream_cancellation(self, pipeline):
        """process_text_stream 被取消后 is_processing 应为 False 且历史不被污染."""

        async def consume_and_cancel():
            gen = pipeline.process_text_stream("测试打断")
            # 获取第一个 chunk 后取消
            chunk = await gen.__anext__()
            assert chunk  # 应至少有一个 chunk
            await gen.aclose()

        await consume_and_cancel()

        # 处理标志应重置
        assert pipeline.is_processing is False
        # 用户消息已在 generator 内添加，但 assistant 消息不应被添加
        # (因为流未完成就被 close)
        user_msgs = [m for m in pipeline._history if m["role"] == "user"]
        assistant_msgs = [m for m in pipeline._history if m["role"] == "assistant"]
        assert len(user_msgs) == 1
        assert len(assistant_msgs) == 0


class TestSessionBargeIn:
    """Session barge-in 集成逻辑测试."""

    @pytest.mark.asyncio
    async def test_barge_in_cancels_processing_task(self):
        """当 voice activity 触发时应取消正在执行的 processing task."""

        # 模拟一个长时间运行的 task
        cancel_observed = False

        async def slow_processing():
            nonlocal cancel_observed
            try:
                await asyncio.sleep(10)  # 模拟长时间处理
            except asyncio.CancelledError:
                cancel_observed = True
                raise

        processing_task = asyncio.create_task(slow_processing())

        # 模拟 barge-in: 取消 task
        await asyncio.sleep(0.05)
        processing_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await processing_task

        assert cancel_observed is True

    @pytest.mark.asyncio
    async def test_barge_in_event_triggers_cancellation(self):
        """模拟完整的 barge-in 流程：event set → 检测 → cancel task."""
        asr = ASRStreamingDashScope(api_key="fake", model="fake")

        # 模拟 processing task
        processing_cancelled = False

        async def mock_processing():
            nonlocal processing_cancelled
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                processing_cancelled = True
                raise

        processing_task = asyncio.create_task(mock_processing())

        # 模拟 barge-in monitor 逻辑
        async def barge_in_monitor():
            await asr.wait_voice_activity()
            if not processing_task.done():
                processing_task.cancel()

        monitor_task = asyncio.create_task(barge_in_monitor())

        # 触发 voice activity
        await asyncio.sleep(0.05)
        asr._voice_activity_event.set()

        # 等待 monitor 执行完毕
        await asyncio.wait_for(monitor_task, timeout=1.0)

        # 等待 processing task 被取消
        with pytest.raises(asyncio.CancelledError):
            await processing_task

        assert processing_cancelled is True
