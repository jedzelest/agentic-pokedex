"""Tool: write an entry to the trainer's journal (Supabase `log_event` table)."""

from typing import Any

from agentic_pokedex.db import log_event
from agentic_pokedex.tools.base import BaseTool

LOG_EVENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string",
                       "enum": ["encounter", "catch", "battle", "milestone", "note"]},
        "region": {"type": "string", "description": "e.g. hoenn, kanto"},
        "location": {"type": "string", "description": "route, cave, gym, city"},
        "pokemon": {"type": "string",
                    "description": "PokéAPI name: lowercase, hyphenated, e.g. mr-mime"},
        "opponent": {"type": "string"},
        "outcome": {"type": "string", "description": "win, loss, caught, fled"},
        "quantity": {"type": "integer", "description": "How many; batch hunts into one row"},
        "is_shiny": {"type": "boolean"},
        "details": {"type": "object", "description": "Extras: held item, moves, level, nature"},
        "note": {"type": "string", "description": "Free-text journal entry"},
    },
    "required": ["event_type"],
}


class LogEventTool(BaseTool):
    """Not a PokéAPI tool: it writes to our own database, so it extends BaseTool directly."""

    name = "log_event"
    description = (
        "Save an event from the user's own playthrough to their trainer journal: "
        "an encounter, a catch, a battle result, a milestone (badge, evolution) or a note. "
        "Only use this when the user tells you something that happened in their game."
    )

    @property
    def parameters(self) -> dict:
        return LOG_EVENT_PARAMETERS

    def run(self, **kwargs: Any) -> dict:
        return {"result": log_event(**kwargs)}
