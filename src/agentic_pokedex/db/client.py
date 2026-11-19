"""Supabase connection, created once and shared (lazily, on first use)."""

import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()
_client: Client | None = None


def is_configured() -> bool:
    """True when the Supabase settings are present (locally in .env, or on Render)."""
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"))


def get_db() -> Client:
    global _client
    if _client is None:
        # The service role key bypasses Row Level Security, so it must stay server-side:
        # never send it to the browser and never commit it.
        _client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    return _client
