"""Keep doc_chunks in sync with the Markdown files in Supabase Storage, and search them.

Each chunk row stores the file's fingerprint (a SHA-256 hash) in its metadata. If the file
in the bucket changes, its fingerprint changes too, so we know it needs re-indexing.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, cast

from postgrest.types import CountMethod

from agentic_pokedex.config import RagSettings
from agentic_pokedex.db import get_db
from agentic_pokedex.rag.chunker import chunk_markdown
from agentic_pokedex.rag.embeddings import Embedder

TABLE = "doc_chunks"


@dataclass
class IndexReport:
    source: str
    status: str  # "indexed", "up to date"
    chunks: int


class NotesIndex:
    def __init__(self, settings: RagSettings | None = None):
        self.settings = settings or RagSettings.from_env()
        self.embedder = Embedder(self.settings)
        self.db = get_db()

    # ---- Storage -----------------------------------------------------------------

    def download(self, source: str) -> str:
        data = self.db.storage.from_(self.settings.bucket).download(source)
        return data.decode("utf-8")

    @staticmethod
    def fingerprint(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def indexed_fingerprint(self, source: str) -> str | None:
        """The fingerprint stored with this file's chunks, or None if it isn't indexed."""
        # Supabase types query results loosely (any JSON), so we tell the type checker the shape.
        rows = cast(list[dict[str, Any]], self.db.table(TABLE).select("metadata")
                    .eq("source", source).limit(1).execute().data)
        metadata = (rows[0].get("metadata") or {}) if rows else {}
        sha = metadata.get("sha256")
        return sha if isinstance(sha, str) else None

    # ---- Indexing ----------------------------------------------------------------

    def reindex(self, source: str, force: bool = False) -> IndexReport:
        """Re-chunk and re-embed one file, replacing its rows. Skips if unchanged."""
        text = self.download(source)
        sha = self.fingerprint(text)
        if not force and self.indexed_fingerprint(source) == sha:
            count = (self.db.table(TABLE).select("id", count=CountMethod.exact, head=True)
                     .eq("source", source).execute().count)
            return IndexReport(source, "up to date", count or 0)

        chunks = chunk_markdown(text)
        vectors = self.embedder.embed([c.content for c in chunks])  # embed first: if this
        rows = [                                                   # fails, old rows stay
            {
                "source": source,
                "chunk_index": c.index,
                "content": c.content,
                "embedding": vector,
                "metadata": {"heading": c.heading, "sha256": sha},
            }
            for c, vector in zip(chunks, vectors)
        ]
        self.db.table(TABLE).delete().eq("source", source).execute()
        self.db.table(TABLE).insert(rows).execute()
        return IndexReport(source, "indexed", len(rows))

    def sync_all(self, force: bool = False) -> list[IndexReport]:
        """Re-index every configured file that changed (or all of them, with force)."""
        return [self.reindex(source, force) for source in self.settings.files]

    @staticmethod
    def sync_on_startup() -> list[str]:
        """Re-index notes that changed in the bucket. Never raises: returns log lines."""
        try:
            return [f"Notes {r.source}: {r.status} ({r.chunks} chunks)"
                    for r in NotesIndex().sync_all()]
        except Exception as exc:  # the app must still start if this fails
            return [f"Notes sync failed, using the existing index: {type(exc).__name__}: {exc}"]

    # ---- Search ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """The chunks closest in meaning to the query (via the match_chunks SQL function)."""
        vector = self.embedder.embed_one(query)
        data = self.db.rpc(
            "match_chunks", {"query_embedding": vector, "match_count": top_k}
        ).execute().data
        return cast(list[dict[str, Any]], data or [])
