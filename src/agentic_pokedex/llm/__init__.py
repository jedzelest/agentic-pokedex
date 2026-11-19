"""Pick the LLM provider from settings. Only the chosen provider's SDK is imported."""

from agentic_pokedex.config import Settings
from agentic_pokedex.llm.base import (
    ChatSession, LLMError, LLMProvider, ModelTurn, ToolCall, ToolResult,
)


def create_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        from agentic_pokedex.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(settings.llm_model)
    if settings.llm_provider == "anthropic":
        from agentic_pokedex.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(settings.llm_model)
    raise SystemExit(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use 'openai' or 'anthropic'.")


__all__ = [
    "ChatSession", "LLMError", "LLMProvider", "ModelTurn", "ToolCall", "ToolResult", "create_llm",
]
