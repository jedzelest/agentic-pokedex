"""Specialization: Pokémon species (lore, evolution, breeding)."""

from agentic_pokedex.tools.base import ResourceTool, english, names


class PokemonSpeciesTool(ResourceTool):
    resource = "pokemon-species"
    name = "get_pokemon_species"
    description = (
        "Get a Pokémon species' Pokédex entry: category, generation, legendary/mythical "
        "status, habitat, what it evolves from, egg groups, capture rate and its forms."
    )

    def summarize(self, data: dict) -> dict:
        evolves_from = data.get("evolves_from_species")
        habitat = data.get("habitat")
        return {
            "name": data["name"],
            "national_dex_number": data["id"],
            "category": english(data["genera"], "genus"),
            "pokedex_entry": english(data["flavor_text_entries"], "flavor_text"),
            "generation": data["generation"]["name"],
            "is_legendary": data["is_legendary"],
            "is_mythical": data["is_mythical"],
            "is_baby": data["is_baby"],
            "evolves_from": evolves_from["name"] if evolves_from else None,
            "habitat": habitat["name"] if habitat else None,
            "color": data["color"]["name"],
            "egg_groups": names(data["egg_groups"]),
            "capture_rate": data["capture_rate"],
            "base_happiness": data["base_happiness"],
            "growth_rate": data["growth_rate"]["name"],
            "forms": names(data["varieties"], key="pokemon"),
        }
