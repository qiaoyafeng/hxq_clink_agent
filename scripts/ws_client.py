"""WebSocket 测试客户端 - 用于本地调试连接天润融通 PCM 语音流 WS Server.

使用方式:
    # 仅连接（不发送音频，但会播放服务端返回的音频）
    uv run python scripts/ws_client.py

    # 连接并实时推送 PCM 文件，同时播放服务端返回的 TTS 音频
    uv run python scripts/ws_client.py --file path/to/audio.pcm

    # 自定义推流参数（默认 4096 bytes/帧, 250ms 间隔 = 16KB/s）
    uv run python scripts/ws_client.py --file audio.pcm --frame-size 4096 --frame-interval 0.25

    # 不播放服务端返回音频
    uv run python scripts/ws_client.py --file audio.pcm --no-playback

或在代码中直接调用 generate_ws_url() 获取完整 URL。
"""

import argparse
import asyncio
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
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


# ---------------------------------------------------------------------------
# PCM 推流
# ---------------------------------------------------------------------------

async def _send_pcm(
    ws,
    file_path: str,
    sample_rate: int = 8000,
    frame_size: int = 4096,
    frame_interval: float = 0.25,
) -> None:
    """读取 PCM 文件并以实时节奏分帧发送，结束后发送 end 信号."""
    pcm_path = Path(file_path)
    if not pcm_path.is_file():
        print(f"[ERROR] PCM 文件不存在: {file_path}")
        return

    file_size = pcm_path.stat().st_size
    bytes_per_sample = 2  # 16-bit PCM
    duration = file_size / (sample_rate * bytes_per_sample)

    print(f"[PCM] 文件       : {pcm_path}")
    print(f"[PCM] 文件大小   : {file_size} bytes")
    print(f"[PCM] 采样率     : {sample_rate} Hz")
    print(f"[PCM] 帧大小     : {frame_size} bytes")
    print(f"[PCM] 帧间隔     : {frame_interval * 1000:.0f} ms")
    print(f"[PCM] 推流速率   : {frame_size / frame_interval / 1024:.1f} KB/s")
    print(f"[PCM] 预计时长   : {duration:.2f} s")
    print(f"[PCM] 开始实时推流...\n")

    sent_frames = 0
    with pcm_path.open("rb") as f:
        start_time = time.monotonic()
        while True:
            chunk = f.read(frame_size)
            if not chunk:
                break
            await ws.send(chunk)
            sent_frames += 1

            # 按实时节奏等待，避免一次性灌入全部数据
            expected_elapsed = sent_frames * frame_interval
            actual_elapsed = time.monotonic() - start_time
            sleep_time = expected_elapsed - actual_elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    print(f"\n[PCM] 推流完成，共发送 {sent_frames} 帧 ({file_size} bytes)")

    # 按协议发送结束信号，通知服务端 flush 缓冲区
    await ws.send(json.dumps({"action": "end"}))
    print("[PCM] 已发送 end 信号\n")


# ---------------------------------------------------------------------------
# 音频播放（sounddevice + 独立线程回调流）
# ---------------------------------------------------------------------------

class AudioPlayer:
    """PCM 音频播放器，使用 sounddevice OutputStream 回调模式.

    服务端返回的 binary 帧被放入队列，由音频回调线程实时消费播放。
    """

    def __init__(self, sample_rate: int = 8000):
        self._sample_rate = sample_rate
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=200)
        self._remainder = b""
        self._stream: sd.OutputStream | None = None
        self._started = False
        self._total_received = 0

    def start(self) -> None:
        """启动音频流."""
        if self._started:
            return
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=int(self._sample_rate * 0.02),  # 20ms per block
            callback=self._callback,
        )
        self._stream.start()
        self._started = True
        print(f"[PLAY] 音频播放已启动 (sample_rate={self._sample_rate} Hz)\n")

    def feed(self, pcm_data: bytes) -> None:
        """将收到的 PCM 帧放入播放队列."""
        self._total_received += len(pcm_data)
        if not self._started:
            self.start()
        try:
            self._queue.put_nowait(pcm_data)
        except queue.Full:
            # 队列满时丢弃最早的帧，防止延迟累积
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(pcm_data)

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice 回调：填充音频输出缓冲区."""
        if status:
            print(f"[PLAY] sounddevice status: {status}")

        bytes_needed = frames * 2  # int16 = 2 bytes/sample

        # 从队列取数据，拼接至所需长度
        buf = self._remainder
        while len(buf) < bytes_needed:
            try:
                chunk = self._queue.get(timeout=0.005)
                if chunk is None:
                    # 收到停止信号
                    self._remainder = b""
                    outdata[:] = 0
                    raise sd.CallbackStop()
                buf += chunk
            except queue.Empty:
                # 数据未到，填充静音
                break

        if len(buf) >= bytes_needed:
            audio_bytes = buf[:bytes_needed]
            self._remainder = buf[bytes_needed:]
        else:
            # 不够一帧，补静音
            audio_bytes = buf + b"\x00" * (bytes_needed - len(buf))
            self._remainder = b""

        outdata[:] = np.frombuffer(audio_bytes, dtype=np.int16).reshape(-1, 1)

    def stop(self) -> None:
        """停止播放."""
        if self._started:
            self._queue.put(None)  # 通知回调线程退出
            # 等待 stream 自然停止
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
            print(f"[PLAY] 音频播放已停止 (共接收 {self._total_received} bytes)")


# ---------------------------------------------------------------------------
# 主连接逻辑
# ---------------------------------------------------------------------------

async def connect_and_listen(
    url: str | None = None,
    pcm_file: str | None = None,
    sample_rate: int = 8000,
    frame_size: int = 4096,
    frame_interval: float = 0.25,
    playback: bool = True,
) -> None:
    """建立 WebSocket 连接，接收服务端消息，并可选地实时推送 PCM 音频和播放回传音频."""
    url = url or generate_ws_url()
    print(f"[INFO] 连接 URL:\n  {url}\n")

    player = AudioPlayer(sample_rate=sample_rate) if playback else None

    async with websockets.connect(url) as ws:
        print("[OK] 连接建立成功，等待服务端消息...\n")

        send_task: asyncio.Task | None = None

        async def _recv_and_dispatch():
            """接收消息：JSON 事件 + binary 音频帧."""
            nonlocal send_task
            async for raw_msg in ws:
                if isinstance(raw_msg, bytes):
                    # 服务端回传的 TTS PCM 音频
                    if player:
                        player.feed(raw_msg)
                    else:
                        print(f"[RECV] binary {len(raw_msg)} bytes (playback disabled)")
                    continue

                msg = json.loads(raw_msg)
                event = msg.get("event", "unknown")
                print(f"[EVENT: {event}] {msg}")

                if event == "started" and pcm_file and send_task is None:
                    print(f"\n[INFO] 会话已启动，sessionId: {msg.get('sessionId')}")
                    print(f"[INFO] 即将推送 PCM 文件: {pcm_file}\n")
                    send_task = asyncio.create_task(
                        _send_pcm(ws, pcm_file, sample_rate, frame_size, frame_interval)
                    )

        recv_task = asyncio.create_task(_recv_and_dispatch())

        try:
            while not recv_task.done():
                tasks = [recv_task]
                if send_task and not send_task.done():
                    tasks.append(send_task)
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                if send_task and send_task.done():
                    print("[INFO] PCM 推流+end 已完成，继续监听服务端消息...\n")
                    send_task = None
        except asyncio.CancelledError:
            pass
        finally:
            recv_task.cancel()
            if send_task and not send_task.done():
                send_task.cancel()
            if player:
                player.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebSocket 测试客户端")
    parser.add_argument("--file", "-f", help="要推送的 PCM 文件路径")
    parser.add_argument(
        "--sample-rate", "-r", type=int, default=8000,
        help="采样率 (默认 8000 Hz)",
    )
    parser.add_argument(
        "--frame-size", "-s", type=int, default=4096,
        help="每帧字节数 (默认 4096 bytes，按协议文档)",
    )
    parser.add_argument(
        "--frame-interval", "-i", type=float, default=0.25,
        help="帧间隔秒数 (默认 0.25s = 250ms，按协议文档)",
    )
    parser.add_argument(
        "--no-playback", action="store_true",
        help="禁用服务端返回音频的本地播放",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        connect_and_listen(
            pcm_file=args.file,
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
            frame_interval=args.frame_interval,
            playback=not args.no_playback,
        )
    )
