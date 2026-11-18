"""The agent's toolbox: hands tool definitions to the LLM and runs the calls it makes."""

import json

from openai.types.chat import ChatCompletionToolUnionParam

from agentic_pokedex.pokeapi import PokeAPIClient, PokeAPINotFound
from agentic_pokedex.tools.ability import AbilityTool
from agentic_pokedex.tools.base import BaseTool
from agentic_pokedex.tools.general import PokeAPILookupTool
from agentic_pokedex.tools.species import PokemonSpeciesTool


class ToolRegistry:
    def __init__(self, tools: list[BaseTool]):
        self._tools = {tool.name: tool for tool in tools}

    @classmethod
    def default(cls, client: PokeAPIClient | None = None) -> "ToolRegistry":
        """The standard Pokédex toolbox: specialized tools first, general fallback last."""
        client = client or PokeAPIClient()
        return cls([
            PokemonSpeciesTool(client),
            AbilityTool(client),
            PokeAPILookupTool(client),
        ])

    def openai_schemas(self) -> list[ChatCompletionToolUnionParam]:
        """All tool definitions, ready for the `tools=` argument of the OpenAI API."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

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
