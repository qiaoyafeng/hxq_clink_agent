"""并发会话上限功能单元测试."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from hxq_clink_agent import ws_server
from hxq_clink_agent.protocol import parse_text_frame


def _make_ws_mock(query_string: bytes = b"uniqueId=test-unique&authString=x") -> MagicMock:
    """构造一个具备最小接口的 WebSocket mock."""
    ws = MagicMock()
    ws.scope = {"query_string": query_string}
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture(autouse=True)
def _reset_sessions_and_auth(monkeypatch):
    """每个用例前后清空 _sessions，并关闭签名校验避免签名依赖."""
    monkeypatch.setattr(ws_server.settings, "auth_enabled", False)
    ws_server._sessions.clear()
    yield
    ws_server._sessions.clear()


class TestConcurrencyLimit:
    """并发上限拒绝路径测试."""

    @pytest.mark.asyncio
    async def test_reject_when_limit_reached(self, monkeypatch):
        """活跃会话数达到上限时应拒绝并发送 |MSG|00|{result:1003}."""
        monkeypatch.setattr(ws_server.settings, "max_concurrent_sessions", 2)
        # 预填 2 个占位会话
        ws_server._sessions["s1"] = MagicMock()
        ws_server._sessions["s2"] = MagicMock()

        ws = _make_ws_mock(b"uniqueId=call-999")
        await ws_server.websocket_endpoint(ws)

        # 应 accept 后发送状态帧再关闭
        ws.accept.assert_awaited_once()
        ws.send_text.assert_awaited_once()
        ws.close.assert_awaited_once()

        # 帧内容应为 |MSG|00|{"result":1003,...}
        sent = ws.send_text.await_args.args[0]
        frame = parse_text_frame(sent)
        assert frame.frame_type == "MSG"
        assert frame.enum_code == "00"
        payload = json.loads(frame.payload)
        assert payload["result"] == 1003
        assert payload["uniqueId"] == "call-999"

        # 关闭码应为 4003
        close_args = ws.close.await_args.args
        assert close_args[0] == 4003

        # _sessions 不应被新增（仍是 2 个占位）
        assert len(ws_server._sessions) == 2
        assert "s1" in ws_server._sessions and "s2" in ws_server._sessions

    @pytest.mark.asyncio
    async def test_unlimited_when_zero(self, monkeypatch):
        """max_concurrent_sessions=0 表示不限制，即使已有活跃会话也放行到 accept."""
        monkeypatch.setattr(ws_server.settings, "max_concurrent_sessions", 0)
        # 预填 100 个占位（远超默认 10），验证 0 是无限制
        for i in range(100):
            ws_server._sessions[f"s{i}"] = MagicMock()

        # 让 Session.run 立即返回，避免真实运行
        async def _noop_run(self):
            return

        monkeypatch.setattr(ws_server.Session, "run", _noop_run)
        # 屏蔽真实 provider 创建，避免网络与 API Key 依赖
        monkeypatch.setattr(ws_server, "create_asr", lambda s: MagicMock())
        monkeypatch.setattr(ws_server, "create_llm", lambda s: MagicMock())
        monkeypatch.setattr(ws_server, "create_tts", lambda s: MagicMock())
        monkeypatch.setattr(ws_server, "create_asr_streaming", lambda s: None)

        ws = _make_ws_mock(b"uniqueId=call-ok")
        await ws_server.websocket_endpoint(ws)

        # 未走拒绝分支：accept 被调用一次，close 未因限流触发（可能因 session 清理未调用）
        ws.accept.assert_awaited_once()
        # 拒绝分支的 send_text 不应命中（限流帧不发送）
        for call in ws.send_text.await_args_list:
            sent = call.args[0]
            if sent.startswith("|MSG|00|"):
                payload = json.loads(parse_text_frame(sent).payload)
                assert payload.get("result") != 1003

    @pytest.mark.asyncio
    async def test_allow_when_under_limit(self, monkeypatch):
        """未达上限时应正常放行到 accept 与 session 创建流程."""
        monkeypatch.setattr(ws_server.settings, "max_concurrent_sessions", 5)
        ws_server._sessions["s1"] = MagicMock()  # 已 1 个，未达上限

        async def _noop_run(self):
            return

        monkeypatch.setattr(ws_server.Session, "run", _noop_run)
        monkeypatch.setattr(ws_server, "create_asr", lambda s: MagicMock())
        monkeypatch.setattr(ws_server, "create_llm", lambda s: MagicMock())
        monkeypatch.setattr(ws_server, "create_tts", lambda s: MagicMock())
        monkeypatch.setattr(ws_server, "create_asr_streaming", lambda s: None)

        ws = _make_ws_mock(b"uniqueId=call-allow")
        await ws_server.websocket_endpoint(ws)

        ws.accept.assert_awaited_once()
        # 不应发出 1003 状态帧
        for call in ws.send_text.await_args_list:
            sent = call.args[0]
            if sent.startswith("|MSG|00|"):
                payload = json.loads(parse_text_frame(sent).payload)
                assert payload.get("result") != 1003
        # session.run 立即返回后 _sessions 应被 finally 清理回 1 个占位
        assert ws_server._sessions == {"s1": ws_server._sessions["s1"]}
