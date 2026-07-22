"""百度RTC大模型互动服务端API客户端.

封装百度云RTC服务端REST API，包括创建/停止智能体实例、打断TTS、发送消息等。
使用 httpx 进行HTTP请求，baidu_bce_auth 模块进行BCE鉴权签名。

API域名: rtc-aiagent.baidubce.com
"""

import json
from dataclasses import dataclass

import httpx
from loguru import logger

from .baidu_bce_auth import build_signed_headers


@dataclass
class InstanceInfo:
    """百度RTC智能体实例信息.

    Attributes:
        instance_id: 大模型互动实例ID
        cid: SDK数据通信ID（用于WebSocket连接参数 id）
        token: SDK数据通信token（用于WebSocket连接参数 t）
        instance_type: 实例类型（如 VoiceChat）
    """

    instance_id: int
    cid: int
    token: str
    instance_type: str = "VoiceChat"


class BaiduRTCAPIClient:
    """百度RTC大模型互动服务端API客户端.

    通过 BCE AK/SK 鉴权调用REST API，管理智能体实例生命周期。
    """

    def __init__(
        self,
        app_id: str,
        ak: str,
        sk: str,
        api_endpoint: str = "rtc-aiagent.baidubce.com",
    ):
        self._app_id = app_id
        self._ak = ak
        self._sk = sk
        self._base_url = f"https://{api_endpoint}"

    def _post(
        self, path: str, body: dict, method: str = "POST"
    ) -> dict:
        """发送带BCE签名的HTTP请求.

        Args:
            path: API路径，如 /api/v1/aiagent/generateAIAgentCall
            body: 请求体字典
            method: HTTP方法

        Returns:
            响应JSON字典

        Raises:
            RuntimeError: API调用失败
        """
        url = f"{self._base_url}{path}"
        body_str = json.dumps(body, ensure_ascii=False)
        headers = build_signed_headers(method, url, body_str, self._ak, self._sk)

        with httpx.Client(timeout=30.0) as client:
            if method == "POST":
                resp = client.post(url, content=body_str, headers=headers)
            elif method == "PUT":
                resp = client.put(url, content=body_str, headers=headers)
            elif method == "DELETE":
                resp = client.delete(url, content=body_str, headers=headers)
            else:
                resp = client.get(url, headers=headers)

        if resp.status_code != 200:
            error_msg = f"BaiduRTC API {path} failed: {resp.status_code} {resp.text}"
            logger.error(error_msg)
            hint = self._auth_error_hint(resp.text)
            if hint:
                logger.error(f"BaiduRTC 鉴权失败排查建议: {hint}")
            raise RuntimeError(error_msg)

        result = resp.json()
        logger.debug(f"BaiduRTC API {path} response: {result}")
        return result

    @staticmethod
    def _auth_error_hint(resp_text: str) -> str:
        """根据 BCE 错误 code 返回可操作的鉴权排查建议.

        区分两类常见 403 鉴权错误：
          - IamSignatureInvalid / SignatureDoesNotMatch：签名不匹配，
            通常是 AK/SK 错误、SK 与 AK 不配对或本机时钟偏差过大。
          - UnauthorizedException：身份已识别但无权限，通常是该 AK/SK
            所属账号/子用户未开通 RTC 大模型互动权限，或 app_id 不属于该账号。

        Args:
            resp_text: 服务端返回的响应体文本

        Returns:
            排查建议字符串；非鉴权类错误时返回空字符串
        """
        try:
            code = json.loads(resp_text).get("code", "")
        except (ValueError, AttributeError):
            return ""

        if "UnauthorizedException" in code:
            return (
                "AK/SK 签名校验通过但无操作权限：请确认 HXQ_BAIDU_RTC_AK/SK "
                "所属账号已开通「RTC 大模型实时互动」服务，且 "
                "HXQ_BAIDU_RTC_APP_ID 属于该账号；若为子用户需在控制台授予 RTC 权限策略。"
            )
        if "IamSignatureInvalid" in code or "SignatureDoesNotMatch" in code:
            return (
                "签名不匹配：请核对 HXQ_BAIDU_RTC_AK 与 HXQ_BAIDU_RTC_SK 是否正确且成对，"
                "并确认运行环境（如 Docker 容器）系统时钟与真实时间同步（时钟偏差过大会导致签名失败）。"
            )
        return ""

    def generate_agent_call(
        self, config: dict | None = None
    ) -> InstanceInfo:
        """创建并启动大模型互动实例.

        Args:
            config: 实例配置字典（角色、TTS、ASR等参数）

        Returns:
            InstanceInfo 实例信息

        Raises:
            RuntimeError: 创建实例失败
        """
        body: dict = {"app_id": self._app_id}
        if config:
            body["config"] = json.dumps(config, ensure_ascii=False)

        logger.info(f"BaiduRTC: generating agent call, config={body.get('config', '')}")

        result = self._post("/api/v1/aiagent/generateAIAgentCall", body)

        context = result.get("context", {})
        instance = InstanceInfo(
            instance_id=result["ai_agent_instance_id"],
            cid=context.get("cid", 0),
            token=context.get("token", ""),
            instance_type=result.get("instance_type", "VoiceChat"),
        )

        logger.info(
            f"BaiduRTC: agent instance created: id={instance.instance_id}, "
            f"cid={instance.cid}, type={instance.instance_type}"
        )
        return instance

    def stop_agent_instance(self, instance_id: int) -> None:
        """停止大模型互动实例，释放资源.

        Args:
            instance_id: 实例ID
        """
        body = {
            "app_id": self._app_id,
            "ai_agent_instance_id": instance_id,
        }
        try:
            self._post("/api/v1/aiagent/stopAIAgentInstance", body)
            logger.info(f"BaiduRTC: agent instance {instance_id} stopped")
        except Exception as e:
            logger.error(f"BaiduRTC: failed to stop instance {instance_id}: {e}")

    def interrupt_tts(
        self, instance_id: int, extra_msg: str = ""
    ) -> None:
        """打断当前TTS播报并播报新内容.

        Args:
            instance_id: 实例ID
            extra_msg: 打断时携带的播报消息
        """
        body: dict = {
            "app_id": self._app_id,
            "ai_agent_instance_id": instance_id,
        }
        if extra_msg:
            body["extra_msg"] = extra_msg
        try:
            self._post("/api/v1/aiagent/interrupt", body)
            logger.info(f"BaiduRTC: TTS interrupted for instance {instance_id}")
        except Exception as e:
            logger.error(f"BaiduRTC: failed to interrupt TTS: {e}")

    def send_message(self, instance_id: int, message: str) -> None:
        """发送消息给端侧（SDK端或WebSocket端）.

        Args:
            instance_id: 实例ID
            message: 自定义消息内容
        """
        body = {
            "app_id": self._app_id,
            "ai_agent_instance_id": instance_id,
            "message": message,
        }
        try:
            self._post("/api/v1/aiagent/sendMsg", body)
            logger.info(f"BaiduRTC: message sent to instance {instance_id}")
        except Exception as e:
            logger.error(f"BaiduRTC: failed to send message: {e}")


def build_agent_config(
    e2e_enabled: bool,
    e2e_prompt: str = "",
    e2e_vcn: int = 8003,
    scene_role_name: str = "",
    scene_role_prompt: str = "",
    tts_vcn: str = "",
    tts_sayhi: str = "",
    lang: str = "zh",
    disable_auto_interrupt: bool = False,
    asr_vad: int = 200,
    audio_codec: str = "raw16k",
    user_id: str = "",
) -> dict:
    """构建百度RTC智能体实例config参数.

    根据模式（端到端/传统托管）生成对应的config字典。

    Args:
        e2e_enabled: 是否启用端到端语音模型
        e2e_prompt: 端到端模型角色提示词
        e2e_vcn: 端到端模型音色
        scene_role_name: 控制台预设角色名
        scene_role_prompt: 临时角色prompt
        tts_vcn: TTS发音人
        tts_sayhi: 招呼语
        lang: 语言
        disable_auto_interrupt: 是否关闭云端自动打断
        asr_vad: ASR断句等待时长(ms)
        audio_codec: 音频编码
        user_id: 用户唯一标识

    Returns:
        config字典
    """
    config: dict = {}

    if e2e_enabled:
        # 端到端语音模型模式
        config["e2ellm_mode"] = "audio_to_audio"
        config["e2ellm_sample_rate"] = 24000
        if e2e_prompt:
            config["e2ellm_prompt"] = e2e_prompt
        config["e2ellm_vcn"] = e2e_vcn
    else:
        # 传统托管模式（ASR-LLM-TTS by Baidu cloud）
        if scene_role_name:
            config["sceneRole"] = scene_role_name
        elif scene_role_prompt:
            config["sceneRoleCfg"] = {
                "name": "custom_role",
                "prompt": scene_role_prompt,
            }

        config["lang"] = lang
        config["tts"] = "DEFAULT"
        if tts_vcn:
            config["tts_url"] = json.dumps(
                {"vcn": tts_vcn}, ensure_ascii=False
            )

        config["asr_vad"] = asr_vad
        config["asr_vad_append"] = True
        config["disable_voice_auto_int"] = disable_auto_interrupt

    # 公共参数
    config["audiocodec"] = audio_codec
    if tts_sayhi:
        config["tts_sayhi"] = tts_sayhi
    if user_id:
        config["user_id"] = user_id

    return config
