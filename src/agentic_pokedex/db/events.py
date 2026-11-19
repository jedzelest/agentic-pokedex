"""Data access for the `log_event` table (the trainer's journal)."""

from typing import Any

from agentic_pokedex.db.client import get_db


def log_event(event_type: str, region: str | None = None, location: str | None = None,
              pokemon: str | None = None, opponent: str | None = None,
              outcome: str | None = None, quantity: int = 1, is_shiny: bool = False,
              details: dict[str, Any] | None = None, note: str | None = None) -> str:
    row = {
        "event_type": event_type, "region": region, "location": location,
        "pokemon": pokemon, "opponent": opponent, "outcome": outcome,
        "quantity": quantity, "is_shiny": is_shiny,
        "details": details or {}, "note": note,
    }
    row = {k: v for k, v in row.items() if v is not None}  # let the table defaults fill gaps
    res = get_db().table("log_event").insert(row).execute()
    return f"Logged event #{res.data[0]['id']} ({event_type})."
