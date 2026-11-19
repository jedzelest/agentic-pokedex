"""Provider-neutral interface between the agent loop and an LLM API.

The agent only ever talks to a ChatSession. Each provider (OpenAI, Anthropic, ...) implements
it in its own message format, so switching providers is a settings change, not a code change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from agentic_pokedex.tools.base import BaseTool


class LLMError(Exception):
    """Any failure talking to the model API (network, auth, rate limit, ...)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class ToolResult:
    call_id: str
    content: str  # JSON string


@dataclass
class ModelTurn:
    """One model reply: some text, and/or a request to run tools."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatSession(ABC):
    """One conversation. Keeps the history in the provider's own native format."""

    @abstractmethod
    def add_user(self, text: str) -> None: ...

    @abstractmethod
    def add_assistant(self, text: str) -> None:
        """Add an earlier assistant reply (used when restoring a conversation)."""

    @abstractmethod
    def complete(self) -> ModelTurn:
        """Send the conversation to the model, record its reply, and return it."""

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None:
        """Send back the results for the tool calls of the last ModelTurn."""


class LLMProvider(ABC):
    """Creates sessions. Holds the API client and model name, so create it once."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def new_session(self, system_prompt: str, tools: list[BaseTool]) -> ChatSession: ...
