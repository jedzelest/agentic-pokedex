"""Application settings, loaded once from the .env file (or the host's environment)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Used when the provider's *_MODEL variable isn't set.
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5"}


@dataclass(frozen=True)
class RagSettings:
    """Where the knowledge notes live and how they are embedded.

    Embeddings always use OpenAI (Anthropic has no embeddings API), independent of
    LLM_PROVIDER. The dimensions must match the doc_chunks.embedding column: vector(1024).
    """
    bucket: str = "RAG Files"
    files: tuple[str, ...] = ("rin_pokemmo_notes.md",)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1024

    @classmethod
    def from_env(cls) -> "RagSettings":
        load_dotenv()
        files = os.getenv("RAG_FILES")
        return cls(
            bucket=os.getenv("RAG_BUCKET", cls.bucket),
            files=tuple(f.strip() for f in files.split(",") if f.strip()) if files else cls.files,
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", cls.embedding_dimensions)),
        )


@dataclass(frozen=True)
class Settings:
    llm_provider: str  # "openai" or "anthropic"
    llm_model: str
    pokeapi_base_url: str = "https://pokeapi.co/api/v2"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        # OPENAI_MODEL or ANTHROPIC_MODEL, so both setups can live side by side in .env.
        model_var = f"{provider.upper()}_MODEL"
        model = os.getenv(model_var) or DEFAULT_MODELS.get(provider)
        if not model:
            raise SystemExit(f"{model_var} is missing from your .env file.")
        # API keys (OPENAI_API_KEY / ANTHROPIC_API_KEY) are read by each SDK itself.
        return cls(llm_provider=provider, llm_model=model)
