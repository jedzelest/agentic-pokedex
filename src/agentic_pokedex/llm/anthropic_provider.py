"""Anthropic implementation (Claude Messages API).

Notes on how Claude differs from OpenAI's Chat Completions:
- The system prompt is a separate `system` argument, not a message.
- Tool calls arrive as `tool_use` content blocks; results go back as `tool_result` blocks,
  all together in ONE user message.
- Recent Claude models think before answering. Their replies contain `thinking` blocks that
  must be sent back unchanged, so we store the reply's full `content`, not just its text.
"""

import json

import anthropic
from anthropic.types.beta import BetaMessageParam, BetaToolParam

from agentic_pokedex.llm.base import (
    ChatSession, LLMError, LLMProvider, ModelTurn, ToolCall, ToolResult,
)
from agentic_pokedex.tools.base import BaseTool

MAX_TOKENS = 16000  # room for thinking + answer; most replies use far less

# If a model's safety check declines a request, these models can hand it to a fallback
# model inside the same call, instead of just stopping. Other models don't support it.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {
    "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1",
}


def _tool_schema(tool: BaseTool) -> BetaToolParam:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    def new_session(self, system_prompt: str, tools: list[BaseTool]) -> ChatSession:
        return AnthropicChatSession(self, system_prompt, tools)


class AnthropicChatSession(ChatSession):
    def __init__(self, provider: AnthropicProvider, system_prompt: str, tools: list[BaseTool]):
        self.provider = provider
        self.system_prompt = system_prompt
        self.tools = [_tool_schema(tool) for tool in tools]
        self.messages: list[BetaMessageParam] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def complete(self) -> ModelTurn:
        model = self.provider.model
        fallback_args = (
            {"betas": [FALLBACK_BETA], "fallbacks": "default"} if model in FALLBACK_MODELS else {}
        )
        try:
            response = self.provider.client.beta.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                messages=self.messages,
                tools=self.tools,
                **fallback_args,
            )
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic: {exc}") from exc

        # Store the full reply (including thinking blocks) so the next request is valid.
        self.messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            return ModelTurn(text="I'm afraid I can't help with that one, Trainer.")

        text = "".join(block.text for block in response.content if block.type == "text")
        calls = [
            ToolCall(block.id, block.name, json.dumps(block.input))
            for block in response.content
            if block.type == "tool_use"
        ]
        if response.stop_reason == "max_tokens" and not text:
            text = "That answer ran too long to finish. Could you narrow the question?"
            calls = []  # a cut-off tool call may have incomplete arguments
        return ModelTurn(text=text, tool_calls=calls)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        # Claude: all results together, in a single user message.
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content}
                for r in results
            ],
        })
