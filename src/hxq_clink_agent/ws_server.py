"""WebSocket 路由 - 接受天润融通客户端连接并处理实时语音流."""

import json
from urllib.parse import parse_qs

from fastapi import APIRouter, WebSocket
from loguru import logger

from .adapters.factory import create_asr, create_asr_streaming, create_llm, create_tts
from .auth import verify_auth
from .config import Settings
from .pipeline import Pipeline
from .protocol import build_session_result
from .session import Session

router = APIRouter()
settings = Settings()

# 活跃会话跟踪
_sessions: dict[str, Session] = {}


def get_active_session_count() -> int:
    """获取当前活跃会话数."""
    return len(_sessions)


@router.websocket(settings.ws_path)
async def websocket_endpoint(ws: WebSocket) -> None:
    """天润融通 PCM 语音流 WebSocket 端点."""

    # 解析查询参数
    query_string = str(ws.scope.get("query_string", b""), "utf-8")
    params = {k: v[0] for k, v in parse_qs(query_string).items()}

    logger.info(f"New WS connection: {params}")

    # 验证签名
    if settings.auth_enabled:
        auth_string = params.get("authString", "")
        if not verify_auth(auth_string, settings.access_key_secret):
            logger.warning(
                f"Auth failed for uniqueId={params.get('uniqueId', params.get('uuid', 'N/A'))}"
            )
            # 发送拒绝消息后关闭连接
            await ws.accept()
            await ws.send_text(
                build_session_result(
                    params.get("uniqueId", params.get("uuid", "")),
                    result=1002,
                    description="无访问权限"
                )
            )
            await ws.close(4001, "Authentication failed")
            return
        logger.debug("Auth verification passed")
    else:
        logger.debug("Auth verification disabled, skipping")

    # 并发上限检查（0 表示不限制）
    max_sessions = settings.max_concurrent_sessions
    if max_sessions > 0 and len(_sessions) >= max_sessions:
        logger.warning(
            f"Reject WS connection: active={len(_sessions)}, limit={max_sessions}, "
            f"uniqueId={params.get('uniqueId', params.get('uuid', 'N/A'))}"
        )
        await ws.accept()
        await ws.send_text(
            build_session_result(
                params.get("uniqueId", params.get("uuid", "")),
                result=1003,
                description="超出最大并发连接数",
            )
        )
        await ws.close(4003, "Max concurrent sessions reached")
        return

    # 接受连接
    await ws.accept()

    # 创建管线和流式 ASR（通过工厂根据配置自动选择 Provider）
    pipeline = Pipeline(
        asr=create_asr(settings),
        llm=create_llm(settings),
        tts=create_tts(settings),
        sample_rate=settings.pcm_sample_rate,
    )
    asr_streaming = create_asr_streaming(settings)

    # 创建并运行会话
    session = Session(
        ws=ws,
        pipeline=pipeline,
        settings=settings,
        params=params,
        asr_streaming=asr_streaming,
    )
    _sessions[session.session_id] = session

    try:
        await session.run()
    finally:
        _sessions.pop(session.session_id, None)
        logger.info(
            f"WS connection closed: sessionId={session.session_id}, "
            f"uniqueId={session.unique_id}"
        )
