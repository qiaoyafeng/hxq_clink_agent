"""百度云 BCE Authorization 签名模块.

实现百度云 OpenAPI 的鉴权签名，基于 HMAC-SHA256。
使用 Python 标准库 hmac + hashlib，无需额外依赖。

签名格式: bce-auth-v1/{accessKeyId}/{timestamp}/{expiration}/{signedHeaders}/{signature}

参考:
  - 官方文档: https://cloud.baidu.com/doc/Reference/s/njwvz1yfu
  - 官方 SDK: https://github.com/baidubce/bce-sdk-python/blob/master/baidubce/auth/bce_v1_signer.py
"""

import base64
import hashlib
import hmac
import uuid
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

# 默认需要签名的头域（除 x-bce-* 外）
_DEFAULT_HEADERS_TO_SIGN = frozenset({
    "host",
    "content-md5",
    "content-length",
    "content-type",
})


def _utc_now_iso() -> str:
    """获取当前 UTC 时间的 ISO 8601 格式字符串."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uri_encode(s: str) -> str:
    """BCE UriEncode: RFC 3986 百分号编码，仅保留非保留字符.

    非保留字符: A-Z a-z 0-9 - _ . ~
    """
    return quote(s, safe="")


def _canonical_uri(path: str) -> str:
    """构建规范化 URI (UriEncodeExceptSlash)."""
    return quote(path, safe="/")


def _canonical_query(query: str) -> str:
    """构建规范化查询字符串.

    按 key 排序，对 key 和 value 分别 UriEncode。
    跳过 authorization 参数。
    """
    if not query:
        return ""
    pairs = []
    for part in query.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
        else:
            k, v = part, ""
        if k.lower() == "authorization":
            continue
        pairs.append((_uri_encode(k), _uri_encode(v)))
    pairs.sort()
    return "&".join(f"{k}={v}" for k, v in pairs)


def _canonical_headers(
    headers: dict[str, str],
    headers_to_sign: set[str] | None = None,
) -> str:
    """构建规范化头域.

    当 headers_to_sign 为 None 时，使用默认头域集（host, content-md5,
    content-length, content-type）。
    所有 x-bce-* 开头的头域始终参与签名。
    对 key 和 value 进行 UriEncode，按 "key:value" 整体字典序排序，用 \n 连接。
    """
    target = headers_to_sign if headers_to_sign is not None else _DEFAULT_HEADERS_TO_SIGN
    result = []
    for k, v in headers.items():
        k_lower = k.strip().lower()
        value = v.strip()
        if not value:
            continue
        if k_lower.startswith("x-bce-") or k_lower in target:
            result.append(f"{_uri_encode(k_lower)}:{_uri_encode(value)}")
    result.sort()
    return "\n".join(result)


def sign_request(
    method: str,
    url: str,
    headers: dict[str, str],
    ak: str,
    sk: str,
    timestamp: str,
    expiration_seconds: int = 1800,
    headers_to_sign: set[str] | None = None,
) -> str:
    """生成百度云 BCE Authorization 签名头.

    遵循百度智能云 BCE v1 签名算法，与官方 Python SDK (bce-sdk-python) 实现一致。

    签名步骤:
      1. authStringPrefix = bce-auth-v1/{ak}/{timestamp}/{expiration}
      2. SigningKey = HMAC-SHA256-HEX(sk, authStringPrefix)  → hex 字符串
      3. CanonicalRequest = method + "\\n" + uri + "\\n" + query + "\\n" + headers
      4. Signature = HMAC-SHA256-HEX(SigningKey, CanonicalRequest)
      5. Authorization = authStringPrefix + "/{signedHeaders}/" + Signature

    Args:
        method: HTTP 方法（POST/GET/PUT/DELETE）
        url: 完整请求 URL
        headers: 请求头字典（必须包含 host）
        ak: Access Key ID
        sk: Secret Access Key
        timestamp: 签名时间戳（必须与 x-bce-date 头域一致）
        expiration_seconds: 签名有效期（秒），默认 1800
        headers_to_sign: 显式指定签名头域集合；为 None 时使用默认集且 signedHeaders 留空

    Returns:
        Authorization 头域值字符串
    """
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path or "/"
    query = parsed.query

    # 确保 headers 中有 host
    all_headers = dict(headers)
    all_headers["host"] = host

    # 1. 构建规范化请求串（仅4部分，不含 signedHeaders 和 body_hash）
    canonical_uri = _canonical_uri(path)
    canonical_query = _canonical_query(query)
    canonical_headers = _canonical_headers(all_headers, headers_to_sign)

    canonical_request = "\n".join([
        method.upper(),
        canonical_uri,
        canonical_query,
        canonical_headers,
    ])

    # 2. 派生签名密钥
    #    authStringPrefix 不含 signedHeaders（与官方 SDK 一致）
    auth_string_prefix = f"bce-auth-v1/{ak}/{timestamp}/{expiration_seconds}"
    signing_key = hmac.new(
        sk.encode("utf-8"),
        auth_string_prefix.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    # 3. 计算签名
    #    注意：signing_key 是 hex 字符串，需 encode 为 bytes 作为第二次 HMAC 的 key
    signature = hmac.new(
        signing_key.encode("utf-8"),
        canonical_request.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    # 4. 构建完整 Authorization 头
    if headers_to_sign:
        signed_headers_str = ";".join(sorted(headers_to_sign))
        return f"{auth_string_prefix}/{signed_headers_str}/{signature}"
    return f"{auth_string_prefix}//{signature}"


def build_signed_headers(
    method: str,
    url: str,
    body: str,
    ak: str,
    sk: str,
    content_type: str = "application/json",
) -> dict[str, str]:
    """构建带 BCE 签名的完整请求头.

    包含 Content-MD5、Content-Length、x-bce-date、x-bce-request-id 等头域，
    确保所有出现在实际 HTTP 请求中的推荐头域均参与签名。

    Args:
        method: HTTP 方法
        url: 完整请求 URL
        body: 请求体字符串（JSON），用于计算 Content-MD5 和 Content-Length
        ak: Access Key ID
        sk: Secret Access Key
        content_type: Content-Type 头值

    Returns:
        包含 Authorization 及所有签名头域的请求头字典
    """
    timestamp = _utc_now_iso()
    host = urlparse(url).netloc
    body_bytes = body.encode("utf-8") if isinstance(body, str) else body

    # 计算请求体的 MD5（Base64），用于防篡改校验
    md5_b64 = base64.b64encode(hashlib.md5(body_bytes).digest()).decode("utf-8")
    content_length = str(len(body_bytes))
    request_id = str(uuid.uuid4())

    # 显式指定参与签名的头域集合（与参考实现一致）
    headers_to_sign = {
        "host",
        "content-md5",
        "content-length",
        "content-type",
        "x-bce-date",
        "x-bce-request-id",
    }

    signing_headers = {
        "host": host,
        "content-type": content_type,
        "content-md5": md5_b64,
        "content-length": content_length,
        "x-bce-date": timestamp,
        "x-bce-request-id": request_id,
    }

    authorization = sign_request(
        method, url, signing_headers, ak, sk, timestamp,
        headers_to_sign=headers_to_sign,
    )

    return {
        "Authorization": authorization,
        "Host": host,
        "Content-Type": content_type,
        "Content-MD5": md5_b64,
        "Content-Length": content_length,
        "x-bce-date": timestamp,
        "x-bce-request-id": request_id,
    }
