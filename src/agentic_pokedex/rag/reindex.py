"""Command line: re-index the knowledge notes.

    uv run python -m agentic_pokedex.rag.reindex           # only files that changed
    uv run python -m agentic_pokedex.rag.reindex --force   # rebuild everything
"""

import sys

from agentic_pokedex.rag.index import NotesIndex


def main() -> None:
    force = "--force" in sys.argv
    for report in NotesIndex().sync_all(force=force):
        print(f"{report.source}: {report.status} ({report.chunks} chunks)")


if __name__ == "__main__":
    main()
