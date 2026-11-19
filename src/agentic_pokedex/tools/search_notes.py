"""Tool: search Rin's PokeMMO knowledge notes (RAG over doc_chunks)."""

from typing import Any

from agentic_pokedex.rag import NotesIndex
from agentic_pokedex.tools.base import BaseTool

MAX_RESULTS = 6


class SearchNotesTool(BaseTool):
    name = "search_notes"
    description = (
        "Search Rin's PokeMMO knowledge notes: regions and progression, gyms and story gates, "
        "breeding, money making, the GTL, EV training spots, shiny hunting, events, PvP rules "
        "and PokeMMO-specific mechanics. Use this for anything specific to PokeMMO; use the "
        "PokéAPI tools for general species, move, ability and type data. Returns the most "
        "relevant note sections, each ending with its sources."
    )

    def __init__(self, index: NotesIndex | None = None):
        self._index = index

    @property
    def index(self) -> NotesIndex:
        if self._index is None:  # connect on first use, not at startup
            self._index = NotesIndex()
        return self._index

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, as a natural question or phrase, "
                                   "e.g. 'best place to EV train Speed in Kanto'.",
                },
                "top_k": {
                    "type": "integer",
                    "description": f"How many sections to return (1-{MAX_RESULTS}, default 4).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def run(self, query: str, top_k: int = 4, **_: Any) -> dict:
        top_k = max(1, min(int(top_k), MAX_RESULTS))
        rows = self.index.search(query, top_k)
        if not rows:
            return {"results": [], "note": "Nothing in the notes matched this query."}
        return {
            "results": [
                {
                    "section": row["content"].split("\n", 1)[0].removeprefix("## ").strip(),
                    "relevance": round(row.get("similarity", 0.0), 3),
                    "source_file": row.get("source"),
                    "text": row["content"],
                }
                for row in rows
            ]
        }
