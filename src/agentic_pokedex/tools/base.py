"""Base classes shared by every tool.

Hierarchy:
    BaseTool          - anything the LLM can call (name, description, parameters, run)
    └── PokeAPITool   - a tool that talks to PokéAPI through a shared client
        └── ResourceTool - a *specialized* tool locked to one resource, with a curated summary
"""

from abc import ABC, abstractmethod
from typing import Any

from openai.types.chat import ChatCompletionFunctionToolParam

from agentic_pokedex.pokeapi import PokeAPIClient


class BaseTool(ABC):
    name: str
    description: str

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema describing the arguments the LLM must send."""

    @abstractmethod
    def run(self, **kwargs: Any) -> dict:
        """Do the work and return JSON-serializable data for the LLM."""

    def to_openai_schema(self) -> ChatCompletionFunctionToolParam:
        """The tool definition in the format OpenAI's Chat Completions API expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class PokeAPITool(BaseTool):
    def __init__(self, client: PokeAPIClient):
        self.client = client


class ResourceTool(PokeAPITool):
    """A specialized tool: fixed resource, takes just a name, returns a clean summary.

    Subclasses set `resource`, `name`, `description` and implement `summarize`.
    """

    resource: str

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": f"Name or ID of the {self.resource}, e.g. 'pikachu' or '25'.",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    def run(self, name: str) -> dict:
        return self.summarize(self.client.get(self.resource, name))

    @abstractmethod
    def summarize(self, data: dict) -> dict:
        """Pick out the useful fields from PokéAPI's (very large) raw response."""


# --- Small helpers the specialized tools share ---------------------------------

def english(entries: list[dict], field: str) -> str | None:
    """Return `field` from the last English entry (PokéAPI lists many languages)."""
    for entry in reversed(entries):
        if entry.get("language", {}).get("name") == "en":
            return " ".join(str(entry[field]).split())  # collapse \n and \f
    return None


def names(items: list[dict], key: str | None = None) -> list[str]:
    """[{'name': 'x', ...}] -> ['x'], optionally reaching into a nested key first."""
    return [(item[key] if key else item)["name"] for item in items]
