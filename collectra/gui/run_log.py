"""Append-only plain-text log of the runs started from the GUI Run button."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from collectra.pipelines.run_plan import RunPlan, RunReport

LOG_NAME = "run-log.txt"
NOTE_LIMIT = 200


def clean_note(note: str) -> str:
    """Collapse all whitespace to single spaces, trim and cap the length."""
    return " ".join(note.split())[:NOTE_LIMIT]


def _append(folder: Path, lines: list[str]) -> None:
    """Append lines to the log; raises OSError if it cannot be written."""
    with open(folder / LOG_NAME, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def begin_entry(
    folder: Path,
    *,
    pipeline: str,
    version: str | None,
    note: str,
    plan: RunPlan,
    now: datetime | None = None,
) -> None:
    """Append the "started" block of a run: when, what, and the plan."""
    now = now or datetime.now()
    title = f"{pipeline} {version}" if version else pipeline
    lines = ["", f"{now:%Y-%m-%d %H:%M:%S}  started  {title}"]
    note = clean_note(note)
    if note:
        lines.append(f"  note: {note}")
    for task in plan.tasks:
        lines.append(f"  {task.name}  planned to {task.action}: {task.reason}")
    _append(folder, lines)


def _outcome_line(name: str, status: str, detail: str) -> str:
    """The one-line description of what happened to a task."""
    detail = " ".join(detail.split())
    if status == "re-ran":
        text = "re-ran, replaced existing output"
        match = re.search(r"upstream (\S+) ran", detail)
        if match:
            text += f" (upstream {match.group(1)} ran)"
    elif status == "skipped":
        text = "skipped (kept existing output)"
    elif status in ("failed", "cancelled"):
        text = f"{status}: {detail}" if detail else status
    else:
        text = status
    return f"  {name}  {text}"


def finish_entry(folder: Path, report: RunReport, seconds: float) -> None:
    """Append one line per task outcome and a summary line."""
    lines = [_outcome_line(o.name, o.status, o.detail) for o in report.outcomes]
    statuses = [outcome.status for outcome in report.outcomes]
    ran = statuses.count("ran") + statuses.count("re-ran")
    summary = (
        f"{ran} ran, {statuses.count('skipped')} skipped,"
        f" {statuses.count('failed')} failed"
    )
    if statuses.count("cancelled"):
        summary += f", {statuses.count('cancelled')} cancelled"
    lines.append(f"  result  {summary}, {seconds:.1f} s")
    _append(folder, lines)
