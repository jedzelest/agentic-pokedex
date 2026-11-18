"""Application settings, loaded once from the .env file."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_model: str
    pokeapi_base_url: str = "https://pokeapi.co/api/v2"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        model = os.getenv("OPENAI_MODEL")
        if not model:
            raise SystemExit("OPENAI_MODEL is missing from your .env file.")
        # OPENAI_API_KEY is read by the OpenAI client itself, so we don't store it here.
        return cls(openai_model=model)
