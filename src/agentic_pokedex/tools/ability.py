"""Specialization: abilities (what they do, who has them)."""

from agentic_pokedex.tools.base import ResourceTool, english

MAX_POKEMON = 30


class AbilityTool(ResourceTool):
    resource = "ability"
    name = "get_ability"
    description = (
        "Get what a Pokémon ability does (in-battle effect) and which Pokémon can have it, "
        "including whether it is their hidden ability."
    )

    def summarize(self, data: dict) -> dict:
        holders = [
            {"pokemon": entry["pokemon"]["name"], "hidden": entry["is_hidden"]}
            for entry in data["pokemon"]
        ]
        return {
            "name": data["name"],
            "generation": data["generation"]["name"],
            "short_effect": english(data["effect_entries"], "short_effect"),
            "effect": english(data["effect_entries"], "effect"),
            "description": english(data["flavor_text_entries"], "flavor_text"),
            "pokemon_with_ability": holders[:MAX_POKEMON],
            "total_pokemon_with_ability": len(holders),
        }
