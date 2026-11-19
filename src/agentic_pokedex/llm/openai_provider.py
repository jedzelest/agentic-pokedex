"""OpenAI implementation (Chat Completions API)."""

import openai
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from agentic_pokedex.llm.base import (
    ChatSession, LLMError, LLMProvider, ModelTurn, ToolCall, ToolResult,
)
from agentic_pokedex.tools.base import BaseTool


def _tool_schema(tool: BaseTool) -> ChatCompletionFunctionToolParam:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = openai.OpenAI()  # reads OPENAI_API_KEY from the environment

    def new_session(self, system_prompt: str, tools: list[BaseTool]) -> ChatSession:
        return OpenAIChatSession(self, system_prompt, tools)


class OpenAIChatSession(ChatSession):
    def __init__(self, provider: OpenAIProvider, system_prompt: str, tools: list[BaseTool]):
        self.provider = provider
        self.tools = [_tool_schema(tool) for tool in tools]
        self.messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def complete(self) -> ModelTurn:
        try:
            response = self.provider.client.chat.completions.create(
                model=self.provider.model,
                messages=self.messages,
                tools=self.tools,
            )
        except openai.APIError as exc:
            raise LLMError(f"OpenAI: {exc}") from exc

        message = response.choices[0].message
        calls = [
            ToolCall(call.id, call.function.name, call.function.arguments)
            for call in message.tool_calls or []
            if call.type == "function"  # we only define function tools
        ]
        if calls:
            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name, "arguments": c.arguments}}
                    for c in calls
                ],
            })
        else:
            self.messages.append({"role": "assistant", "content": message.content or ""})
        return ModelTurn(text=message.content or "", tool_calls=calls)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for result in results:  # OpenAI: one "tool" message per call
            self.messages.append({
                "role": "tool", "tool_call_id": result.call_id, "content": result.content,
            })
