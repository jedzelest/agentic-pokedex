import sys

import openai

from agentic_pokedex.agent import PokedexAgent
from agentic_pokedex.config import Settings
from agentic_pokedex.pokeapi import PokeAPIClient
from agentic_pokedex.tools import ToolRegistry


def main() -> None:
    """Interactive Pokédex in the terminal. Run with:  uv run agentic-pokedex"""
    sys.stdout.reconfigure(encoding="utf-8")  # so "Pokémon" prints correctly on Windows

    settings = Settings.from_env()
    registry = ToolRegistry.default(PokeAPIClient(settings.pokeapi_base_url))
    agent = PokedexAgent(model=settings.openai_model, registry=registry)

    print(f"Rin, your Pokédex assistant ({settings.openai_model}). Ask anything about Pokémon.")
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
        except openai.APIError as exc:
            print(f"(OpenAI error: {exc})\n")
            continue
        print(f"Rin: {answer}\n")

    print("Goodbye!")
