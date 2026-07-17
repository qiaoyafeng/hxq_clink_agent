"""天润融通三方机器人语音接入协议 - 帧编解码模块.

帧格式定义：
- 控制帧: |CTL|{enum}|{payload}
- 状态帧: |MSG|{enum}|{payload}
- 数据帧(上行音频): |DAT|00|{binary_pcm}   (8字节头)
- 数据帧(下行音频): |DAT|01|{资源类型}|{key}|{end_flag}|{binary_pcm}
"""

import json
from dataclasses import dataclass

# ── 帧头常量 ──
DAT_UPLINK_HEADER = b"|DAT|00|"  # 上行音频帧头 (8 bytes)
DAT_DOWNLINK_PREFIX = b"|DAT|01|"  # 下行音频帧前缀 (8 bytes)

RESOURCE_TYPE_AUDIO = "audio/wav"


@dataclass
class TextFrame:
    """解析后的文本帧."""

    frame_type: str  # "CTL", "MSG", or "UNKNOWN"
    enum_code: str  # "00", "01", "02", "03", "04" etc.
    payload: str  # JSON 或空字符串


@dataclass
class BinaryFrame:
    """解析后的二进制帧."""

    frame_type: str  # "DAT_00" (uplink audio) or "UNKNOWN"
    payload: bytes  # PCM 数据


# ── 解析函数 ──


def parse_binary_frame(data: bytes) -> BinaryFrame:
    """解析二进制帧，提取帧类型和 PCM payload.

    Args:
        data: 原始二进制 WebSocket 消息

    Returns:
        BinaryFrame，frame_type 为 "DAT_00" 表示上行音频
    """
    if data[:8] == DAT_UPLINK_HEADER:
        return BinaryFrame(frame_type="DAT_00", payload=data[8:])
    # 兼容旧协议：无帧头的裸 PCM 数据
    return BinaryFrame(frame_type="RAW_PCM", payload=data)


def parse_text_frame(text: str) -> TextFrame:
    """解析文本帧，提取帧类型、enum 和 payload.

    支持格式：
    - |CTL|00|{json}
    - |MSG|00|{json}
    - |MSG|02|
    - |CTL|02|
    - |CTL|04|audio/wav|{key}|INT|

    Args:
        text: 原始文本 WebSocket 消息

    Returns:
        TextFrame，若不匹配已知格式则 frame_type 为 "UNKNOWN"
    """
    if not text.startswith("|"):
        return TextFrame(frame_type="UNKNOWN", enum_code="", payload=text)

    # 拆分: ["", "CTL"/"MSG"/"DAT", "enum", "payload..."]
    # 最多切 3 段以保留 payload 中可能的 '|'
    parts = text.split("|", 3)
    if len(parts) < 3:
        return TextFrame(frame_type="UNKNOWN", enum_code="", payload=text)

    frame_type = parts[1]  # "CTL", "MSG", "DAT"
    enum_code = parts[2]  # "00", "01", "02", "03", "04"
    payload = parts[3] if len(parts) > 3 else ""

    if frame_type in ("CTL", "MSG", "DAT"):
        return TextFrame(frame_type=frame_type, enum_code=enum_code, payload=payload)

    return TextFrame(frame_type="UNKNOWN", enum_code="", payload=text)


# ── 构造函数 ──


def build_msg_frame(enum_code: str, payload_dict: dict | None = None) -> str:
    """构造状态帧文本.

    Args:
        enum_code: 状态码枚举，如 "00", "01", "02"
        payload_dict: JSON 负载字典，None 则无负载

    Returns:
        格式化的状态帧文本，如 |MSG|00|{"result":1000,...}
    """
    if payload_dict is not None:
        return f"|MSG|{enum_code}|{json.dumps(payload_dict, ensure_ascii=False)}"
    return f"|MSG|{enum_code}|"


def build_ctl_frame(enum_code: str, payload: str = "") -> str:
    """构造控制帧文本.

    Args:
        enum_code: 控制码枚举，如 "02" (挂机), "03" (转人工)
        payload: 文本负载，如 JSON 字符串

    Returns:
        格式化的控制帧文本，如 |CTL|02| 或 |CTL|03|{"qno":"9999"}
    """
    return f"|CTL|{enum_code}|{payload}"


def build_dat_downlink(key: int, is_end: bool, pcm: bytes) -> bytes:
    """构造下行音频数据帧（二进制）.

    格式: |DAT|01|audio/wav|{key}|{end_flag}|{pcm_bytes}

    Args:
        key: 资源 key（0 起始正整数，由小到大表示播放顺序）
        is_end: 该资源是否已结束（True=最后一帧）
        pcm: PCM 音频数据

    Returns:
        完整的二进制帧（header + PCM）
    """
    end_flag = "1" if is_end else "0"
    header = f"|DAT|01|{RESOURCE_TYPE_AUDIO}|{key}|{end_flag}|".encode("ascii")
    return header + pcm


def build_resource_control(key: int) -> str:
    """构造资源控制帧（打断/中断客户端播放）.

    客户端收到后应清除所有 key <= 指定 key 的缓存并停止播放。

    Args:
        key: 要中断的资源 key 上限

    Returns:
        格式化的控制帧文本，如 |CTL|04|audio/wav|3|INT|
    """
    return f"|CTL|04|{RESOURCE_TYPE_AUDIO}|{key}|INT|"


def build_session_result(unique_id: str, result: int = 1000, description: str = "OK") -> str:
    """构造 session 开启结果状态帧.

    Args:
        unique_id: 通话唯一标识
        result: 状态码（1000=成功）
        description: 状态描述

    Returns:
        |MSG|00|{"result":1000,"description":"OK","uniqueId":"..."}
    """
    return build_msg_frame("00", {
        "result": result,
        "description": description,
        "uniqueId": unique_id,
    })
