"""Name the results folder an input file is imported into."""

from __future__ import annotations

from collections.abc import Callable
from itertools import count
from pathlib import Path

from pydantic import BaseModel, Field


def destination(source: Path, ext: str, taken: Callable[[Path], bool]) -> Path:
    """The first results folder name for ``source`` that is not taken.

    The candidates sit next to the source: ``<stem>.<ext>``, then
    ``<stem>-2.<ext>``, ``<stem>-3.<ext>`` and so on.

    Args:
        source: The input file.
        ext: The pipeline's results folder extension, without the dot.
        taken: Whether a candidate cannot be used.

    Returns:
        The first candidate that is not taken.
    """
    for number in count(1):
        suffix = f"-{number}" if number > 1 else ""
        candidate = source.parent / f"{source.stem}{suffix}.{ext}"
        if not taken(candidate):
            return candidate


class ImportReport(BaseModel):
    """What opening a set of inputs did.

    Attributes:
        pages: Every results folder created or opened, in the order processed.
        imported: How many of them were created.
        existing: How many of them were already results folders.
        skipped: ``"<name>: <reason>"`` for each file the pipeline does not accept.
        skipped_folders: ``"<name>: <reason>"`` for each ``<x>.<ext>`` folder
            that could not be adopted.
        failed: ``"<name>: <reason>"`` for each input that could not be imported.
    """

    pages: list[Path] = Field(default_factory=list)
    imported: int = 0
    existing: int = 0
    skipped: list[str] = Field(default_factory=list)
    skipped_folders: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)

    def summary(self, name: str) -> str:
        """A one-line notice of what was done.

        Args:
            name: The pipeline's name.

        Returns:
            The notice, or an empty string when only existing pages were opened.
        """

        def plural(count: int, noun: str) -> str:
            return f"{count} {noun}" + ("" if count == 1 else "s")

        parts = []
        if self.imported:
            parts.append(f"Imported {plural(self.imported, 'page')}")
        if self.existing:
            parts.append(f"opened {self.existing} existing")
        if self.skipped:
            parts.append(
                f"skipped {plural(len(self.skipped), 'file')} {name} doesn't accept"
            )
        if self.skipped_folders:
            parts.append(
                f"skipped {plural(len(self.skipped_folders), 'folder')} "
                "without exactly one image"
            )
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if not (self.imported or self.skipped or self.skipped_folders or self.failed):
            return ""
        return ", ".join(parts)
