"""WebSocket 路由 - 接受天润融通客户端连接并处理实时语音流."""

import json
from urllib.parse import parse_qs

from fastapi import APIRouter, WebSocket
from loguru import logger

from .adapters import (
    ASRDashScope,
    ASRStreamingDashScope,
    ASRStub,
    LLMOpenAI,
    LLMStub,
    TTSDashScope,
    TTSStub,
)
from .auth import verify_auth
from .config import Settings
from .pipeline import Pipeline
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
                f"Auth failed for uniqueId={params.get('uniqueId', 'N/A')}"
            )
            # 发送拒绝消息后关闭连接
            await ws.accept()
            await ws.send_text(
                json.dumps({"event": "error", "message": "拒绝建立连接：签名验证失败"})
            )
            await ws.close(4001, "Authentication failed")
            return
        logger.debug("Auth verification passed")
    else:
        logger.debug("Auth verification disabled, skipping")

    # 接受连接
    await ws.accept()

    # 创建管线和流式 ASR（根据配置选择 Stub 或真实适配器）
    asr_streaming = None
    if settings.use_stub:
        pipeline = Pipeline(
            asr=ASRStub(),
            llm=LLMStub(),
            tts=TTSStub(),
            sample_rate=settings.pcm_sample_rate,
        )
    else:
        pipeline = Pipeline(
            asr=ASRDashScope(
                api_key=settings.dashscope_api_key,
                model=settings.asr_model,
            ),
            llm=LLMOpenAI(
                api_key=settings.dashscope_api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                system_prompt=settings.llm_system_prompt,
            ),
            tts=TTSDashScope(
                api_key=settings.dashscope_api_key,
                base_url=settings.tts_base_url,
                model=settings.tts_model,
                voice=settings.tts_voice,
            ),
            sample_rate=settings.pcm_sample_rate,
        )

        # 流式 ASR（仅非 Stub 模式且配置启用时创建）
        if settings.asr_streaming_enabled:
            asr_streaming = ASRStreamingDashScope(
                api_key=settings.dashscope_api_key,
                model=settings.asr_model,
                sample_rate=settings.pcm_sample_rate,
                max_sentence_silence=settings.asr_max_sentence_silence,
            )

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
