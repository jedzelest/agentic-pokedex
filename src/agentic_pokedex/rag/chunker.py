"""Split a Markdown knowledge file into chunks, one per `##` section."""

import re
from dataclasses import dataclass

MAX_WORDS = 600  # a section longer than this is split by paragraph


@dataclass
class Chunk:
    index: int  # position in the file (doc_chunks.chunk_index)
    heading: str  # e.g. "Breeding: IV inheritance with Power items"
    content: str  # heading + section text: what gets embedded and what Rin reads


def chunk_markdown(text: str) -> list[Chunk]:
    """One chunk per `## ` section. Text before the first `##` (title, byline) is skipped.

    Every chunk starts with its section heading, so it makes sense on its own and the
    heading's words help the search find it.
    """
    sections = re.split(r"^## +", text, flags=re.MULTILINE)[1:]  # [0] = title/preamble
    chunks: list[Chunk] = []
    for section in sections:
        heading, _, body = section.partition("\n")
        heading = heading.strip()
        for part in _split_long(body.strip()):
            chunks.append(Chunk(len(chunks), heading, f"## {heading}\n\n{part}"))
    return chunks


def _split_long(body: str) -> list[str]:
    """Keep a section whole unless it's over MAX_WORDS; then group paragraphs into pieces."""
    if len(body.split()) <= MAX_WORDS:
        return [body]
    pieces, current = [], []
    for paragraph in re.split(r"\n\s*\n", body):
        if current and len(" ".join(current + [paragraph]).split()) > MAX_WORDS:
            pieces.append("\n\n".join(current))
            current = []
        current.append(paragraph)
    if current:
        pieces.append("\n\n".join(current))
    return pieces
