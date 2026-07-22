"""百度RTC WebSocket 连接诊断探针.

独立复现「连接后立即被百度断开」问题，捕获精确的 WS 关闭码/原因，
并尝试主动发送 License 激活，判断根因。

用法:
    uv run python scripts/baidu_probe.py
"""

import asyncio
import json
import time

import websockets
from websockets.asyncio.client import connect

from hxq_clink_agent.adapters.baidu_rtc_client import (
    BaiduRTCAPIClient,
    build_agent_config,
)
from hxq_clink_agent.config import Settings

_settings = Settings()


async def probe(unique_device: bool) -> None:
    s = _settings
    device_id = s.baidu_rtc_device_id
    user_id = s.baidu_rtc_user_id
    if unique_device:
        # 用唯一 device/user 规避「多设备共用 license」检测
        suffix = str(time.time_ns())
        device_id = f"{device_id}-{suffix}"
        user_id = f"{user_id}-{suffix}"

    print("=" * 70)
    print(f"probe: unique_device={unique_device}")
    print(f"  device_id = {device_id}")
    print(f"  user_id   = {user_id}")
    print(f"  license   = {s.baidu_rtc_license_key}")
    print("=" * 70)

    config = build_agent_config(
        e2e_enabled=s.baidu_rtc_e2e_enabled,
        e2e_prompt=s.baidu_rtc_e2e_prompt,
        e2e_vcn=s.baidu_rtc_e2e_vcn,
        scene_role_name=s.baidu_rtc_scene_role_name,
        scene_role_prompt=s.baidu_rtc_scene_role_prompt,
        tts_vcn=s.baidu_rtc_tts_vcn,
        tts_sayhi=s.baidu_rtc_tts_sayhi,
        lang=s.baidu_rtc_lang,
        disable_auto_interrupt=s.baidu_rtc_disable_auto_interrupt,
        asr_vad=s.baidu_rtc_asr_vad,
        audio_codec=s.baidu_rtc_audio_codec,
        user_id=user_id,
    )

    api = BaiduRTCAPIClient(
        app_id=s.baidu_rtc_app_id,
        ak=s.baidu_rtc_ak,
        sk=s.baidu_rtc_sk,
        api_endpoint=s.baidu_rtc_api_endpoint,
    )

    loop = asyncio.get_running_loop()
    instance = await loop.run_in_executor(None, api.generate_agent_call, config)
    print(f"[API] instance created: id={instance.instance_id}, cid={instance.cid}")

    ws_url = (
        f"{s.baidu_rtc_ws_endpoint}"
        f"?a={s.baidu_rtc_app_id}"
        f"&id={instance.instance_id}"
        f"&t={instance.token}"
        f"&ac={s.baidu_rtc_audio_codec}"
    )
    print(f"[WS] connecting: {ws_url}\n")

    lic_sent = False
    try:
        async with connect(ws_url) as ws:
            print("[WS] connected, waiting for messages...\n")
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
                except asyncio.TimeoutError:
                    print("[WS] 15s elapsed, no more messages")
                    break

                if isinstance(msg, bytes):
                    print(f"[WS] << BINARY {len(msg)} bytes")
                    continue

                print(f"[WS] << TEXT: {msg}")
                if msg.startswith("[E]:[LIC]:[MUST]") and not lic_sent:
                    lic = (
                        f'[E]:[LIC]:[ACTIVE]:{{"devId":"{device_id}",'
                        f'"uId":"{user_id}",'
                        f'"licKey":"{s.baidu_rtc_license_key}"}}'
                    )
                    print(f"[WS] >> sending license activation: {lic}")
                    await ws.send(lic)
                    lic_sent = True
    except websockets.ConnectionClosed as e:
        print(f"\n[WS] CONNECTION CLOSED: code={e.code} reason={e.reason!r}")
        print(f"[WS] rcvd={getattr(e, 'rcvd', None)} sent={getattr(e, 'sent', None)}")
    except Exception as e:
        print(f"\n[WS] ERROR: {type(e).__name__}: {e}")
    finally:
        await loop.run_in_executor(None, api.stop_agent_instance, instance.instance_id)
        print("[API] instance stopped\n")


async def main() -> None:
    # 先用固定 device/user 复现问题
    await probe(unique_device=False)
    await asyncio.sleep(2)
    # 再用唯一 device/user 验证是否为「license 共用」导致
    await probe(unique_device=True)


if __name__ == "__main__":
    asyncio.run(main())
