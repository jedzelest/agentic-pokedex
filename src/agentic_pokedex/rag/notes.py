"""The notes file as a list of `##` sections, so Rin can change one section at a time."""

import difflib
import re
from dataclasses import dataclass
from datetime import date

HEADING = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)
RIN_MARKER = re.compile(r"\n*^_(?:Added|Edited) by Rin on [^_\n]+\._\s*$", re.MULTILINE)


class NoteEditError(ValueError):
    """A change that can't be made (unknown section, bad heading, ...). Shown to the model."""


@dataclass
class Section:
    heading: str  # without the "## "
    body: str  # everything under the heading, up to the next "## "


def _key(heading: str) -> str:
    """Compare headings loosely: case, extra spaces and leading #'s don't matter."""
    return " ".join(heading.lstrip("#").split()).casefold()


class NotesDocument:
    def __init__(self, preamble: str, sections: list[Section]):
        self.preamble = preamble  # title and byline, before the first "##"
        self.sections = sections

    # ---- Reading and writing the file ------------------------------------------

    @classmethod
    def parse(cls, text: str) -> "NotesDocument":
        matches = list(HEADING.finditer(text))
        if not matches:
            return cls(text, [])
        sections = []
        for n, match in enumerate(matches):
            end = matches[n + 1].start() if n + 1 < len(matches) else len(text)
            sections.append(Section(match.group(1).strip(), text[match.end():end].strip("\n")))
        return cls(text[:matches[0].start()], sections)

    def render(self) -> str:
        parts = [self.preamble.rstrip("\n")] if self.preamble.strip() else []
        parts += [f"## {s.heading}\n\n{s.body}" for s in self.sections]
        return "\n\n".join(parts) + "\n"

    # ---- Finding sections ------------------------------------------------------

    def find(self, heading: str) -> int:
        """Index of the section with this heading, or a NoteEditError with suggestions."""
        wanted = _key(heading)
        for n, section in enumerate(self.sections):
            if _key(section.heading) == wanted:
                return n
        close = difflib.get_close_matches(
            heading, [s.heading for s in self.sections], n=3, cutoff=0.4)
        hint = f" Did you mean: {'; '.join(close)}?" if close else ""
        raise NoteEditError(f"No section called '{heading}'.{hint}")

    # ---- Changing sections -----------------------------------------------------

    def replace(self, heading: str, new_body: str) -> tuple[Section, Section]:
        """Replace a section's body. Returns (before, after)."""
        n = self.find(heading)
        before = self.sections[n]
        after = Section(before.heading, _with_marker(new_body, "Edited"))
        self.sections[n] = after
        return before, after

    def add_after(self, related_heading: str, heading: str, body: str) -> Section:
        """Insert a new section right after a related one."""
        heading = " ".join(heading.lstrip("#").split())
        if not heading or len(heading) > 120:
            raise NoteEditError("The new section needs a heading of at most 120 characters.")
        if any(_key(s.heading) == _key(heading) for s in self.sections):
            raise NoteEditError(f"A section called '{heading}' already exists; replace it instead.")
        n = self.find(related_heading)
        section = Section(heading, _with_marker(body, "Added"))
        self.sections.insert(n + 1, section)
        return section


def _with_marker(body: str, verb: str) -> str:
    """Clean the body and end it with "_Added/Edited by Rin on <date>._" (only one marker)."""
    body = RIN_MARKER.sub("", body.strip("\n")).rstrip()
    if not body:
        raise NoteEditError("The section text is empty.")
    for line in body.splitlines():
        if re.match(r"^#{1,2} ", line):  # "# x" or "## x" would break the section structure
            raise NoteEditError(
                f"The text contains a top-level heading ('{line[:60]}'). Use plain text or "
                "'###' sub-headings inside a section.")
    return f"{body}\n\n_{verb} by Rin on {date.today().isoformat()}._"
