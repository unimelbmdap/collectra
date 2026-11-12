"""The LLM tokens and cost of one pipeline run, added up by task."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from decimal import Decimal

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


@dataclass
class TaskUsage:
    """What one task's LLM calls used during a run.

    ``input_tokens`` includes the tokens read from and written to the cache.
    ``cost`` is in US dollars and covers only the calls that could be priced;
    ``unpriced`` counts the others, such as calls to a local model.
    """

    model: str
    calls: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: Decimal = Decimal(0)
    unpriced: int = 0

    def add(self, usage) -> None:
        self.calls += 1
        self.requests += usage.requests
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_write_tokens += usage.cache_write_tokens
        if usage.cost is None:
            self.unpriced += 1
        else:
            self.cost += usage.cost


def _dollars(cost: Decimal, calls: int, unpriced: int) -> str:
    if calls and unpriced == calls:
        return "unknown"
    amount = f"${cost:,.2f}" if cost >= 1 else f"${cost:.4f}"
    return f"≥ {amount}" if unpriced else amount


class UsageLedger:
    """Adds up each task's LLM usage over a run. Safe to add to from any thread."""

    def __init__(self) -> None:
        self.tasks: dict[str, TaskUsage] = {}
        self._lock = threading.Lock()

    def __bool__(self) -> bool:
        return bool(self.tasks)

    def add(self, task: str, model: str, usage) -> None:
        """Add one call's usage (a pydantic-ai ``RunUsage``) to ``task``'s total."""
        with self._lock:
            self.tasks.setdefault(task, TaskUsage(model=model)).add(usage)

    def table(self) -> Table:
        table = Table(
            title="LLM usage",
            title_style="bold",
            box=box.SIMPLE_HEAD,
            pad_edge=False,
            show_edge=False,
            collapse_padding=True,
            caption="Input includes the cached tokens. Costs are estimates in US dollars.",
        )
        # No cell wraps; two-line headings keep the table within 80 columns.
        table.add_column("Task", style="bold", no_wrap=True)
        table.add_column("Model", no_wrap=True)
        for heading in ("Calls", "Input", "Cache\nread", "Cache\nwrite", "Output"):
            table.add_column(heading, justify="right", no_wrap=True)
        table.add_column("Cost", justify="right", style="green", no_wrap=True)

        total = TaskUsage(model="")
        for name, entry in self.tasks.items():
            table.add_row(*self._cells(Text(name), Text(entry.model), entry))
            for field in (
                "calls",
                "requests",
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "cost",
                "unpriced",
            ):
                setattr(total, field, getattr(total, field) + getattr(entry, field))
        table.add_section()
        table.add_row(*self._cells(Text("Total"), Text(""), total))
        return table

    @staticmethod
    def _cells(name: Text, model: Text, entry: TaskUsage) -> list:
        return [
            name,
            model,
            f"{entry.calls:,}",
            f"{entry.input_tokens:,}",
            f"{entry.cache_read_tokens:,}",
            f"{entry.cache_write_tokens:,}",
            f"{entry.output_tokens:,}",
            _dollars(entry.cost, entry.calls, entry.unpriced),
        ]

    def print_summary(self, console: Console | None = None) -> None:
        """Print the table, or nothing when no LLM was called."""
        if not self:
            return
        console = console or Console(highlight=False)
        console.print(self.table())
        unpriced = sum(entry.unpriced for entry in self.tasks.values())
        if unpriced:
            calls, are = ("call", "is") if unpriced == 1 else ("calls", "are")
            console.print(
                Text.assemble(
                    ("! ", "yellow"),
                    f"{unpriced:,} {calls} could not be priced and {are} "
                    "not in the cost.",
                )
            )
