"""The generalization: one tool that can reach any PokéAPI resource."""

import json
from typing import Any

from agentic_pokedex.pokeapi import PokeAPIClient
from agentic_pokedex.tools.base import PokeAPITool

MAX_LIST_ITEMS = 15  # e.g. a Pokémon's 100+ moves get cut to 15
MAX_CHARS = 8000  # cap on what we send back to the LLM
# Fields that are big but rarely useful for answering questions (image URLs, per-game indices).
NOISY_FIELDS = {"sprites", "cries", "game_indices", "version_group_details", "machines"}


class PokeAPILookupTool(PokeAPITool):
    name = "pokeapi_lookup"
    description = (
        "Look up data from any PokéAPI resource (Pokémon, moves, items, types, "
        "berries, locations, ...). Use this when no specialized tool fits. "
        "Long lists are shortened and very large fields may be omitted."
    )

    def __init__(self, client: PokeAPIClient):
        super().__init__(client)
        # The main endpoint tells us every resource that exists - one call, cached.
        self.resources = client.list_resources()

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "resource": {
                    "type": "string",
                    "enum": self.resources,
                    "description": "Which PokéAPI resource to query.",
                },
                "name": {
                    "type": "string",
                    "description": "Name or ID of the entry, e.g. 'thunderbolt' or '85'.",
                },
            },
            "required": ["resource", "name"],
            "additionalProperties": False,
        }

    def run(self, resource: str, name: str) -> dict:
        data = self._shrink(self.client.get(resource, name))
        return self._fit(data)

    def _shrink(self, value: Any) -> Any:
        """Recursively remove noise: non-English entries, noisy fields, long lists."""
        if isinstance(value, dict):
            return {k: self._shrink(v) for k, v in value.items() if k not in NOISY_FIELDS}
        if isinstance(value, list):
            value = [item for item in value if not _is_foreign_language(item)]
            kept = [self._shrink(item) for item in value[:MAX_LIST_ITEMS]]
            if len(value) > MAX_LIST_ITEMS:
                kept.append(f"... {len(value) - MAX_LIST_ITEMS} more omitted")
            return kept
        return value

    @staticmethod
    def _fit(data: dict) -> dict:
        """If still too big, drop the largest fields whole, so the JSON stays valid."""
        omitted = []
        while len(json.dumps(data)) > MAX_CHARS and len(data) > 1:
            largest = max(data, key=lambda key: len(json.dumps(data[key])))
            data = {k: v for k, v in data.items() if k != largest}
            omitted.append(largest)
        if omitted:
            data["omitted_fields_too_large"] = omitted
        return data


def _is_foreign_language(item: Any) -> bool:
    """True for PokéAPI translation entries in a language other than English."""
    if isinstance(item, dict) and isinstance(item.get("language"), dict):
        return item["language"].get("name") != "en"
    return False
