"""FastAPI 应用实例 - 统一管理 HTTP 和 WebSocket 端点."""

from fastapi import FastAPI

from .config import Settings
from .health import router as health_router
from .ws_server import router as ws_router

settings = Settings()

_docs_url = None if settings.production else "/docs"
_redoc_url = None if settings.production else "/redoc"
_openapi_url = None if settings.production else "/openapi.json"

app = FastAPI(
    title="hxq-clink-agent",
    description="天润融通 PCM 语音流实时推送 Server (ASR-LLM-TTS)",
    version="0.1.0",
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    openapi_url=_openapi_url,
)

app.include_router(health_router)
app.include_router(ws_router)
