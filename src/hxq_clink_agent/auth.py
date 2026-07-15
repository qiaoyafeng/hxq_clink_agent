"""鉴权模块 - 验证天润融通客户端 WebSocket 连接的 authString 签名."""

import base64
import hashlib
import hmac
from urllib.parse import quote, unquote


def _build_base_string(app_id: str, access_key_id: str, timestamp: str) -> str:
    """构造 baseString：将三个字段用逗号拼接后 URL encode.

    文档规则：appId,accessKeyId,timestamp → url_encode
    """
    raw = f"{app_id},{access_key_id},{timestamp}"
    # 按文档示例，逗号被编码为 %2C，冒号被编码为 %3A 等
    return quote(raw, safe="")


def _compute_signature(secret_key: str, base_string: str) -> str:
    """使用 HMAC-SHA1 对 baseString 签名并 base64 编码."""
    digest = hmac.new(
        secret_key.encode("utf-8"),
        base_string.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def generate_signature(
    app_id: str,
    access_key_id: str,
    timestamp: str,
    secret_key: str,
) -> str:
    """生成签名（可用于测试或客户端模拟）."""
    base_string = _build_base_string(app_id, access_key_id, timestamp)
    return _compute_signature(secret_key, base_string)


def generate_auth_string(
    app_id: str,
    access_key_id: str,
    timestamp: str,
    secret_key: str,
) -> str:
    """生成完整 authString（baseString + signature，逗号拼接后 url_encode）.

    格式: url_encode("appId,accessKeyId,timestamp,signature")
    """
    signature = generate_signature(app_id, access_key_id, timestamp, secret_key)
    raw_auth = f"{app_id},{access_key_id},{timestamp},{signature}"
    return quote(raw_auth, safe="")


def verify_auth(auth_string: str, secret_key: str) -> bool:
    """验证客户端发来的 authString.

    步骤:
    1. URL decode authString
    2. 按逗号分割 → [appId, accessKeyId, timestamp, signature]
    3. 用前三段重建 baseString 并 url_encode
    4. 用 secret_key 对 baseString 做 HMAC-SHA1 + base64
    5. 比对计算结果与客户端传来的 signature
    """
    try:
        decoded = unquote(auth_string)
        parts = decoded.split(",")
        if len(parts) < 4:
            return False

        app_id = parts[0]
        access_key_id = parts[1]
        timestamp = parts[2]
        client_signature = ",".join(parts[3:])  # signature 中可能不含逗号，但保险起见

        base_string = _build_base_string(app_id, access_key_id, timestamp)
        expected_signature = _compute_signature(secret_key, base_string)

        return hmac.compare_digest(expected_signature, client_signature)
    except Exception:
        return False
