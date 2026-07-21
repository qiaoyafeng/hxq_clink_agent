"""转人工（Transfer to Agent）功能单元测试."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from hxq_clink_agent.config import Settings
from hxq_clink_agent.protocol import build_transfer_frame, parse_text_frame
from hxq_clink_agent.session import Session


class TestBuildTransferFrame:
    """protocol.build_transfer_frame 单元测试."""

    def test_build_transfer_frame_format(self):
        """构造帧应为 |CTL|03|{"qno":"9999"} 格式."""
        frame = build_transfer_frame("9999")
        assert frame.startswith("|CTL|03|")

        parsed = parse_text_frame(frame)
        assert parsed.frame_type == "CTL"
        assert parsed.enum_code == "03"
        payload = json.loads(parsed.payload)
        assert payload == {"qno": "9999"}

    def test_build_transfer_frame_custom_qno(self):
        """qno 应可自定义."""
        frame = build_transfer_frame("1234")
        parsed = parse_text_frame(frame)
        payload = json.loads(parsed.payload)
        assert payload["qno"] == "1234"

    def test_build_transfer_frame_unicode_qno(self):
        """qno 支持非 ASCII 字符（保留中文而非 \\uxxxx 转义）."""
        frame = build_transfer_frame("客服组A")
        parsed = parse_text_frame(frame)
        payload = json.loads(parsed.payload)
        assert payload["qno"] == "客服组A"


def _make_session(
    *,
    transfer_enabled: bool = True,
    transfer_keywords: str = "转人工,人工客服,人工服务",
    transfer_qno: str = "9999",
) -> Session:
    """构造一个仅供 _detect_transfer / _trigger_transfer 测试用的 Session 实例.

    绕开 __init__ 的复杂依赖，直接注入必要属性。
    """
    settings = Settings(
        transfer_enabled=transfer_enabled,
        transfer_keywords=transfer_keywords,
        transfer_qno=transfer_qno,
        auth_enabled=False,
        dashscope_api_key="",
    )

    session = Session.__new__(Session)
    session.session_id = "test-session"
    session._settings = settings
    session._ws = MagicMock()
    session._ws.send_text = AsyncMock()
    session._closed = False
    session._transferring = False
    session._processing_task = None
    session._resource_key = 0
    if settings.transfer_enabled:
        session._transfer_keywords = [
            kw.strip()
            for kw in settings.transfer_keywords.split(",")
            if kw.strip()
        ]
    else:
        session._transfer_keywords = []
    return session


class TestDetectTransfer:
    """Session._detect_transfer 单元测试."""

    def test_detects_exact_keyword(self):
        session = _make_session()
        assert session._detect_transfer("转人工") is True

    def test_detects_keyword_within_sentence(self):
        session = _make_session()
        assert session._detect_transfer("我要转人工，谢谢") is True

    def test_detects_alternative_keyword(self):
        session = _make_session()
        assert session._detect_transfer("请帮我接人工客服") is True

    def test_miss_when_no_keyword(self):
        session = _make_session()
        assert session._detect_transfer("你好，今天天气怎么样？") is False

    def test_empty_text_returns_false(self):
        session = _make_session()
        assert session._detect_transfer("") is False

    def test_disabled_returns_false(self):
        """transfer_enabled=False 时任何文本都不命中."""
        session = _make_session(transfer_enabled=False)
        assert session._transfer_keywords == []
        assert session._detect_transfer("转人工") is False

    def test_empty_keywords_returns_false(self):
        """keywords 为空字符串时不命中."""
        session = _make_session(transfer_keywords="")
        assert session._transfer_keywords == []
        assert session._detect_transfer("转人工") is False

    def test_keywords_are_trimmed(self):
        """关键词两侧空白应被去掉，仍能命中."""
        session = _make_session(transfer_keywords="  转人工  ,  找人工 ")
        assert "转人工" in session._transfer_keywords
        assert "找人工" in session._transfer_keywords
        assert session._detect_transfer("我想转人工") is True


class TestTriggerTransfer:
    """Session._trigger_transfer 集成流程测试."""

    @pytest.mark.asyncio
    async def test_trigger_sends_resource_control_and_transfer_frame(self):
        """命中后应先发资源控制帧再发 |CTL|03|，并标记 _transferring=True."""
        session = _make_session(transfer_qno="8888")

        await session._trigger_transfer("我要转人工")

        assert session._transferring is True
        # 应至少发送 2 条 text 消息：资源控制帧 + 转人工帧
        assert session._ws.send_text.await_count == 2

        sent_frames = [call.args[0] for call in session._ws.send_text.await_args_list]
        # 首个是资源控制帧
        assert sent_frames[0].startswith("|CTL|04|")
        # 第二个是转人工帧
        transfer_frame = sent_frames[1]
        assert transfer_frame.startswith("|CTL|03|")
        parsed = parse_text_frame(transfer_frame)
        payload = json.loads(parsed.payload)
        assert payload["qno"] == "8888"

    @pytest.mark.asyncio
    async def test_trigger_cancels_processing_task(self):
        """存在正在进行的 processing task 时应被取消."""
        import asyncio

        session = _make_session()

        async def long_running():
            await asyncio.sleep(10)

        task = asyncio.create_task(long_running())
        session._processing_task = task

        await session._trigger_transfer("找人工")

        # 允许事件循环处理取消
        await asyncio.sleep(0)
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_trigger_marks_closed_on_send_failure(self):
        """发送转人工帧异常时应立即置 _closed=True."""
        session = _make_session()

        # 让第 2 次 send_text（转人工帧）抛异常
        session._ws.send_text = AsyncMock(
            side_effect=[None, RuntimeError("network broken")]
        )

        await session._trigger_transfer("人工服务")

        assert session._transferring is True
        assert session._closed is True
