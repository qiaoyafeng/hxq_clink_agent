"""LLM 适配器 - 通过 OpenAI 兼容接口调用大语言模型."""

import httpx
from loguru import logger

from ..interfaces.llm import LLMInterface


class LLMOpenAI(LLMInterface):
    """OpenAI 兼容接口 LLM 适配器.

    支持任何 OpenAI Chat Completions 兼容的 API 端点，
    包括 DashScope（通义千问）、DeepSeek、OpenAI 等。

    Args:
        api_key: API Key
        base_url: API 基础地址，如 https://dashscope.aliyuncs.com/compatible-mode/v1
        model: 模型名称，如 qwen-turbo
        system_prompt: 系统提示词
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-turbo",
        system_prompt: str = "你是一个智能语音助手，请用简洁的语言回答用户的问题。",
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._system_prompt = system_prompt

    async def chat(self, text: str, history: list[dict[str, str]]) -> str:
        """根据用户输入和对话历史生成回复.

        构造 OpenAI Chat Completions 格式的请求，包含 system prompt
        和对话历史，通过 HTTP POST 发送至兼容端点。
        """
        messages: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt}]
        messages.extend(history)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": messages,
                    },
                )
                response.raise_for_status()
                data = response.json()

            content = data["choices"][0]["message"]["content"]
            logger.debug(f"LLM reply: {content!r}")
            return content

        except httpx.HTTPStatusError as e:
            logger.error(
                f"LLM OpenAI HTTP error: status={e.response.status_code}, "
                f"body={e.response.text[:200]}"
            )
            return ""
        except Exception as e:
            logger.error(f"LLM OpenAI exception: {e}")
            return ""
