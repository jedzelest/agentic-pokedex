"""RAG: Rin's searchable knowledge notes (Markdown in Supabase Storage -> doc_chunks)."""

from agentic_pokedex.rag.chunker import Chunk, chunk_markdown
from agentic_pokedex.rag.editor import NotesEditor
from agentic_pokedex.rag.index import IndexReport, NotesIndex
from agentic_pokedex.rag.notes import NoteEditError, NotesDocument

__all__ = [
    "Chunk", "IndexReport", "NoteEditError", "NotesDocument", "NotesEditor", "NotesIndex",
    "chunk_markdown",
]
