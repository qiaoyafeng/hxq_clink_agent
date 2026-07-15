"""健康检查路由 - 提供容器健康检查、存活探针等 HTTP 端点."""

from collections.abc import Callable

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["health"])

# 用于从外部注入活跃会话数获取函数
_get_session_count: Callable[[], int] = lambda: 0


def set_session_count_getter(getter: Callable[[], int]) -> None:
    """设置活跃会话数获取函数（由 WSServer 调用注入）."""
    global _get_session_count
    _get_session_count = getter


@router.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    """存活探针：容器健康检查调用此端点."""
    return "OK"


@router.get("/ready", response_class=PlainTextResponse)
async def ready() -> str:
    """就绪探针：确认服务已就绪可接受流量."""
    return "OK"


@router.get("/status")
async def status() -> dict:
    """详细状态：返回当前活跃会话数等信息."""
    return {
        "status": "running",
        "active_sessions": _get_session_count(),
    }
