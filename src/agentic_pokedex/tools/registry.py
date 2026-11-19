"""The agent's toolbox: hands tool definitions to the LLM and runs the calls it makes."""

import json
import os

from agentic_pokedex import db
from agentic_pokedex.pokeapi import PokeAPIClient, PokeAPINotFound
from agentic_pokedex.tools.ability import AbilityTool
from agentic_pokedex.tools.base import BaseTool
from agentic_pokedex.tools.general import PokeAPILookupTool
from agentic_pokedex.tools.log_event import LogEventTool
from agentic_pokedex.tools.search_notes import SearchNotesTool
from agentic_pokedex.tools.species import PokemonSpeciesTool


class ToolRegistry:
    def __init__(self, tools: list[BaseTool], notices: list[str] | None = None):
        self._tools = {tool.name: tool for tool in tools}
        self.notices = notices or []  # which optional features are on/off, for startup logs

    @classmethod
    def default(cls, client: PokeAPIClient | None = None) -> "ToolRegistry":
        """The standard Pokédex toolbox: specialized tools first, general fallback last."""
        client = client or PokeAPIClient()
        tools: list[BaseTool] = [
            PokemonSpeciesTool(client),
            AbilityTool(client),
            PokeAPILookupTool(client),
        ]
        notices: list[str] = []
        # Optional features need Supabase (and the notes need OpenAI for embeddings).
        # Without their settings the app still runs, minus that feature - and says so.
        if db.is_configured():
            tools.append(LogEventTool())
            notices.append("Journal logging: ON")
        else:
            notices.append("Journal logging: OFF (set SUPABASE_URL and SUPABASE_SERVICE_KEY)")
        if db.is_configured() and os.getenv("OPENAI_API_KEY"):
            tools.append(SearchNotesTool())
            notices.append("Knowledge notes search: ON")
        else:
            notices.append("Knowledge notes search: OFF (needs Supabase settings and "
                           "OPENAI_API_KEY for embeddings)")
        return cls(tools, notices)

    @property
    def tools(self) -> list[BaseTool]:
        """All tools. Each LLM provider converts them into its own API format."""
        return list(self._tools.values())

    def execute(self, name: str, arguments_json: str) -> str:
        """Run a tool call from the LLM and return the result as a JSON string.

        Errors are returned as data (not raised) so the LLM can read them and recover,
        e.g. by fixing a misspelled Pokémon name.
        """
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool '{name}'."})
        try:
            result = tool.run(**json.loads(arguments_json))
        except PokeAPINotFound:
            result = {"error": "Not found in PokéAPI. Check the spelling or try another name."}
        except Exception as exc:  # network errors, bad arguments, ...
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, ensure_ascii=False)
