"""Tools: let Rin change the knowledge notes, always as propose -> trainer confirms -> apply."""

from typing import Any

from agentic_pokedex.rag.editor import NotesEditor
from agentic_pokedex.rag.notes import NoteEditError
from agentic_pokedex.tools.base import BaseTool


class ProposeNoteEditTool(BaseTool):
    name = "propose_note_edit"
    description = (
        "Draft a change to Rin's PokeMMO knowledge notes: replace one section's text, or add a "
        "new section right after the most related existing one. Nothing is saved: this returns "
        "a preview and a draft_id. Show the trainer the change and ask them to confirm. Only "
        "use this when the trainer asks to add to or correct the notes (playthrough events go "
        "to log_event instead)."
    )

    def __init__(self, editor: NotesEditor):
        self.editor = editor

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["replace", "add"]},
                "section_heading": {
                    "type": "string",
                    "description": "replace: the exact heading of the section to rewrite. "
                                   "add: the heading of the NEW section, in the notes' "
                                   "'Topic: subtopic' style, e.g. 'Breeding: Egg move tips'.",
                },
                "new_text": {
                    "type": "string",
                    "description": "The full section body (without the '## heading' line). "
                                   "For replace, include everything that should remain. Keep "
                                   "the notes' style and end with a 'Source:' line if known.",
                },
                "after_section": {
                    "type": "string",
                    "description": "add only: heading of the most related existing section; "
                                   "the new section is placed right after it.",
                },
            },
            "required": ["action", "section_heading", "new_text"],
            "additionalProperties": False,
        }

    def run(self, action: str, section_heading: str, new_text: str,
            after_section: str | None = None, **_: Any) -> dict:
        try:
            if action == "replace":
                preview = self.editor.propose_replace(section_heading, new_text)
            elif action == "add":
                if not after_section:
                    return {"error": "For 'add', set after_section to the most related "
                                     "existing section's heading."}
                preview = self.editor.propose_add(after_section, section_heading, new_text)
            else:
                return {"error": "action must be 'replace' or 'add'."}
        except NoteEditError as exc:
            return {"error": str(exc)}
        preview["next_step"] = ("Not saved yet. Show the trainer this change and ask whether to "
                                "save it. Call apply_note_edit only after they say yes.")
        return preview


class ApplyNoteEditTool(BaseTool):
    name = "apply_note_edit"
    description = (
        "Save a drafted notes change (from propose_note_edit) after the trainer has clearly "
        "said yes. Backs up the current file, saves the draft and re-indexes the notes."
    )
    not_in_same_reply_as = "propose_note_edit"  # the trainer must answer in between

    def __init__(self, editor: NotesEditor):
        self.editor = editor

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "draft_id": {"type": "string", "description": "The draft_id from the proposal."},
            },
            "required": ["draft_id"],
            "additionalProperties": False,
        }

    def run(self, draft_id: str, **_: Any) -> dict:
        try:
            return self.editor.apply(draft_id)
        except NoteEditError as exc:
            return {"error": str(exc)}
