import sys

from agentic_pokedex.agent import PokedexAgent
from agentic_pokedex.config import Settings
from agentic_pokedex.llm import LLMError, create_llm
from agentic_pokedex.pokeapi import PokeAPIClient
from agentic_pokedex.tools import ToolRegistry


def main() -> None:
    """Interactive Pokédex in the terminal. Run with:  uv run agentic-pokedex"""
    sys.stdout.reconfigure(encoding="utf-8")  # so "Pokémon" prints correctly on Windows

    settings = Settings.from_env()
    registry = ToolRegistry.default(PokeAPIClient(settings.pokeapi_base_url))
    agent = PokedexAgent(llm=create_llm(settings), registry=registry)

    for notice in registry.notices:
        print(notice)
    if registry.get("search_notes"):
        from agentic_pokedex.rag import NotesIndex
        for line in NotesIndex.sync_on_startup():
            print(line)
    print(f"\nRin, your Pokédex assistant ({settings.llm_provider}: {settings.llm_model}). "
          "Ask anything about Pokémon.")
    print("Commands: 'reset' to start a new conversation, 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break
        if question.lower() == "reset":
            agent.reset()
            print("(conversation cleared)\n")
            continue

        try:
            answer = agent.ask(question)
        except LLMError as exc:
            print(f"(Model error: {exc})\n")
            continue
        print(f"Rin: {answer}\n")

    print("Goodbye!")
