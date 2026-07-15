"""LLM 占位适配器 - 开发/测试用."""

from ..interfaces.llm import LLMInterface


class LLMStub(LLMInterface):
    """LLM 占位实现，回显输入文本，用于开发联调."""

    async def chat(self, text: str, history: list[dict[str, str]]) -> str:
        """返回包含输入的占位回复."""
        return f"收到：{text}"
