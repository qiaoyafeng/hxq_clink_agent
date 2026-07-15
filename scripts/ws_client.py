"""WebSocket 测试客户端 - 用于本地调试连接天润融通 PCM 语音流 WS Server.

使用方式:
    uv run python scripts/ws_client.py

或在代码中直接调用 generate_ws_url() 获取完整 URL。
"""

import asyncio
import json
import time

import websockets

from hxq_clink_agent.config import Settings

# 从 .env / 环境变量统一加载，与 WS Server 使用同一份配置
_settings = Settings()


def generate_ws_url(
    host: str = "localhost",
    port: int | None = None,
    ws_path: str | None = None,
    app_id: str | None = None,
    access_key_id: str | None = None,
    access_key_secret: str | None = None,
    unique_id: str = "test-001",
    cno: str = "1001",
) -> str:
    """生成带鉴权参数的完整 WebSocket URL.

    默认值均从 Settings() 读取，无需手动维护。
    返回示例:
        ws://localhost:18000/realtime_voice?uniqueId=test-001&cno=1001&authString=...
    """
    from hxq_clink_agent.auth import generate_auth_string

    # 未传入时使用 .env 中的配置
    port = port or _settings.port
    ws_path = ws_path or _settings.ws_path
    app_id = app_id or _settings.app_id
    access_key_id = access_key_id or _settings.access_key_id
    access_key_secret = access_key_secret or _settings.access_key_secret

    timestamp = str(int(time.time()))
    auth_string = generate_auth_string(app_id, access_key_id, timestamp, access_key_secret)

    # 打印各参数明细
    print("=" * 60)
    print("WebSocket 连接参数")
    print("=" * 60)
    print(f"  host              : {host}")
    print(f"  port              : {port}")
    print(f"  ws_path           : {ws_path}")
    print(f"  uniqueId          : {unique_id}")
    print(f"  cno               : {cno}")
    print(f"  appId             : {app_id}")
    print(f"  accessKeyId       : {access_key_id}")
    print(f"  timestamp         : {timestamp}")
    print(f"  accessKeySecret   : {access_key_secret}")
    print(f"  authString (编码) : {auth_string}")
    print("=" * 60)

    url = (
        f"ws://{host}:{port}{ws_path}"
        f"?uniqueId={unique_id}"
        f"&cno={cno}"
        f"&authString={auth_string}"
    )
    print(f"\n完整 URL:\n  {url}\n")
    return url


async def connect_and_listen(url: str | None = None) -> None:
    """建立 WebSocket 连接并打印服务端消息."""
    url = url or generate_ws_url()
    print(f"[INFO] 连接 URL:\n  {url}\n")

    async with websockets.connect(url) as ws:
        print("[OK] 连接建立成功，等待服务端消息...\n")
        async for raw_msg in ws:
            msg = json.loads(raw_msg)
            event = msg.get("event", "unknown")
            print(f"[EVENT: {event}] {msg}")

            # 收到 started 事件后，可在此处发送 PCM 音频帧
            if event == "started":
                print("\n[INFO] 会话已启动，sessionId:", msg.get("sessionId"))
                print("[INFO] 如需发送 PCM 数据，请调用 ws.send(bytes_data)\n")
                # 示例：发送一帧静音 PCM（全零），实际使用时替换为真实音频
                # silence_frame = b"\x00" * 4096
                # await ws.send(silence_frame)


if __name__ == "__main__":
    asyncio.run(connect_and_listen())
