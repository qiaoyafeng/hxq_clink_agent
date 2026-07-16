"""LLM 抽象接口."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator


class LLMInterface(ABC):
    """大语言模型抽象基类."""

    @abstractmethod
    async def chat(self, text: str, history: list[dict[str, str]]) -> str:
        """根据用户输入和对话历史生成回复.

        Args:
            text: 当前用户输入文本
            history: 对话历史，格式 [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            模型生成的回复文本
        """
        ...

    @abstractmethod
    async def chat_stream(
        self, text: str, history: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """流式生成回复，逐 token yield 文本片段.

        Args:
            text: 当前用户输入文本
            history: 对话历史，格式 [{"role": "user"|"assistant", "content": "..."}]

        Yields:
            模型生成的文本片段（token）
        """
        ...
        # 使其成为 async generator
        yield  # type: ignore  # pragma: no cover
