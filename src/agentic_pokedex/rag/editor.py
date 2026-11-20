"""Propose-then-apply editing of the notes file in Supabase Storage.

1. propose_*  -> builds the full new file in memory and keeps it as a *draft*. Nothing saved.
2. apply()    -> after the trainer said yes: backup -> save that exact draft -> re-index.
"""

import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath

from agentic_pokedex.rag.index import NotesIndex
from agentic_pokedex.rag.notes import NoteEditError, NotesDocument

DRAFT_TTL_SECONDS = 15 * 60  # an unconfirmed draft expires after 15 minutes
CONTENT_TYPE = "text/markdown; charset=utf-8"


@dataclass
class Draft:
    id: str
    source: str
    base_sha: str  # fingerprint of the file the draft was made from
    new_text: str  # the complete new file
    summary: str
    created: float = field(default_factory=time.time)


class NotesEditor:
    def __init__(self, index: NotesIndex):
        self.index = index
        self._drafts: dict[str, Draft] = {}
        self._lock = threading.Lock()  # the web app can serve several chats at once

    @property
    def source(self) -> str:
        return self.index.settings.files[0]  # the main notes file

    # ---- Step 1: propose -----------------------------------------------------------

    def propose_replace(self, heading: str, new_body: str) -> dict:
        text, doc = self._load()
        before, after = doc.replace(heading, new_body)
        draft = self._store(text, doc, f"Replace section '{before.heading}'")
        return {
            "draft_id": draft.id,
            "action": "replace",
            "section": before.heading,
            "before": before.body,
            "after": after.body,
        }

    def propose_add(self, related_heading: str, heading: str, body: str) -> dict:
        text, doc = self._load()
        section = doc.add_after(related_heading, heading, body)
        related = doc.sections[doc.find(section.heading) - 1].heading
        draft = self._store(text, doc, f"Add section '{section.heading}' after '{related}'")
        return {
            "draft_id": draft.id,
            "action": "add",
            "section": section.heading,
            "placed_after": related,
            "new_section": f"## {section.heading}\n\n{section.body}",
        }

    # ---- Step 2: apply -------------------------------------------------------------

    def apply(self, draft_id: str) -> dict:
        with self._lock:
            self._expire()
            draft = self._drafts.pop(draft_id, None)
        if draft is None:
            raise NoteEditError("That draft doesn't exist or has expired. Propose the change again.")

        current = self.index.download(draft.source)
        if self.index.fingerprint(current) != draft.base_sha:
            raise NoteEditError("The notes file changed since this draft was made (maybe a new "
                                "upload). Propose the change again from the current version.")

        bucket = self.index.db.storage.from_(self.index.settings.bucket)
        backup = self._backup_path(draft.source)
        # Text must be sent as bytes: a plain str would be treated as a file path.
        bucket.upload(backup, current.encode("utf-8"), {"content-type": CONTENT_TYPE})
        bucket.update(draft.source, draft.new_text.encode("utf-8"),
                      {"content-type": CONTENT_TYPE, "upsert": "true"})
        report = self.index.reindex(draft.source)
        return {
            "saved": True,
            "change": draft.summary,
            "backup": backup,
            "reindexed_chunks": report.chunks,
        }

    # ---- Helpers -------------------------------------------------------------------

    def _load(self) -> tuple[str, NotesDocument]:
        text = self.index.download(self.source)
        return text, NotesDocument.parse(text)

    def _store(self, base_text: str, doc: NotesDocument, summary: str) -> Draft:
        draft = Draft(
            id=secrets.token_hex(4),
            source=self.source,
            base_sha=self.index.fingerprint(base_text),
            new_text=doc.render(),
            summary=summary,
        )
        with self._lock:
            self._expire()
            self._drafts[draft.id] = draft
        return draft

    def _expire(self) -> None:
        cutoff = time.time() - DRAFT_TTL_SECONDS
        for draft_id in [d.id for d in self._drafts.values() if d.created < cutoff]:
            del self._drafts[draft_id]

    @staticmethod
    def _backup_path(source: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = PurePosixPath(source)
        return f"history/{name.stem}.{stamp}{name.suffix}"
