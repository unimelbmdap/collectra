"""Name the results folder an input file is imported into."""

from __future__ import annotations

from collections.abc import Callable
from itertools import count
from pathlib import Path


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
