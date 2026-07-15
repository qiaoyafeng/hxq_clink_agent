"""鉴权模块单元测试."""

import pytest

from hxq_clink_agent.auth import (
    generate_auth_string,
    generate_signature,
    verify_auth,
    _build_base_string,
)


class TestBuildBaseString:
    """测试 baseString 构造."""

    def test_basic_encoding(self):
        """基本 URL 编码测试."""
        result = _build_base_string("tr001", "clink2", "2018-09-25T19:34:42+0800")
        # 逗号应被编码为 %2C，冒号为 %3A，加号为 %2B
        assert "%2C" in result
        assert "tr001" in result
        assert "clink2" in result

    def test_special_chars_encoded(self):
        """特殊字符应被正确编码."""
        result = _build_base_string("app1", "key1", "2024-01-01T00:00:00+0800")
        # + 应被编码为 %2B
        assert "%2B0800" in result
        # : 应被编码为 %3A
        assert "%3A" in result


class TestGenerateSignature:
    """测试签名生成."""

    def test_signature_not_empty(self):
        """生成的签名不应为空."""
        sig = generate_signature("tr001", "clink2", "2018-09-25T19:34:42+0800", "secret123")
        assert sig
        assert len(sig) > 0

    def test_signature_deterministic(self):
        """相同输入应产生相同签名."""
        sig1 = generate_signature("app", "key", "ts", "secret")
        sig2 = generate_signature("app", "key", "ts", "secret")
        assert sig1 == sig2

    def test_different_secret_different_signature(self):
        """不同密钥应产生不同签名."""
        sig1 = generate_signature("app", "key", "ts", "secret1")
        sig2 = generate_signature("app", "key", "ts", "secret2")
        assert sig1 != sig2


class TestVerifyAuth:
    """测试签名验证."""

    def test_valid_auth_string(self):
        """正确的 authString 应验证通过."""
        auth = generate_auth_string(
            "tr001", "clink2", "2018-09-25T19:34:42+0800", "mysecret"
        )
        assert verify_auth(auth, "mysecret") is True

    def test_invalid_secret(self):
        """错误的密钥应验证失败."""
        auth = generate_auth_string(
            "tr001", "clink2", "2018-09-25T19:34:42+0800", "mysecret"
        )
        assert verify_auth(auth, "wrongsecret") is False

    def test_tampered_auth_string(self):
        """篡改的 authString 应验证失败."""
        auth = generate_auth_string(
            "tr001", "clink2", "2018-09-25T19:34:42+0800", "mysecret"
        )
        # 篡改：替换部分字符
        tampered = auth[:-5] + "XXXXX"
        assert verify_auth(tampered, "mysecret") is False

    def test_empty_auth_string(self):
        """空 authString 应验证失败."""
        assert verify_auth("", "mysecret") is False

    def test_malformed_auth_string(self):
        """格式错误的 authString 应验证失败."""
        assert verify_auth("not,enough,parts", "mysecret") is False
        assert verify_auth("garbage", "mysecret") is False
