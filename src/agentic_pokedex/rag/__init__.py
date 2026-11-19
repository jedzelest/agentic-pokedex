"""RAG: Rin's searchable knowledge notes (Markdown in Supabase Storage -> doc_chunks)."""

from agentic_pokedex.rag.chunker import Chunk, chunk_markdown
from agentic_pokedex.rag.index import IndexReport, NotesIndex

__all__ = ["Chunk", "IndexReport", "NotesIndex", "chunk_markdown"]
