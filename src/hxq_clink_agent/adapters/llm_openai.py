"""LLM 适配器 - 通过 OpenAI 兼容接口调用大语言模型."""

import json
import time
from collections.abc import AsyncGenerator

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
            t_start = time.monotonic()
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
            elapsed = time.monotonic() - t_start
            logger.info(f"LLM reply ({elapsed:.2f}s): {content!r}")
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

    async def chat_stream(
        self, text: str, history: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """流式生成回复，通过 SSE 逐 token yield 文本片段.

        使用 OpenAI Chat Completions 的 stream=true 模式，
        解析 SSE 事件流并逐步 yield delta.content。
        """
        messages: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt}]
        messages.extend(history)

        try:
            t_start = time.monotonic()
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": messages,
                        "stream": True,
                    },
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        logger.error(
                            f"LLM OpenAI stream HTTP error: status={response.status_code}, "
                            f"body={body.decode('utf-8', errors='replace')[:200]}"
                        )
                        return

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            logger.debug(f"LLM stream parse skip: {e}")
                            continue

            elapsed = time.monotonic() - t_start
            logger.info(f"LLM stream completed ({elapsed:.2f}s)")

        except Exception as e:
            logger.error(f"LLM OpenAI stream exception: {e}")
