"""A run collects every task's LLM usage, and the terminal run prints a summary."""

from decimal import Decimal
from pathlib import Path

from pydantic_ai.usage import RunUsage

from collectra.types.images import Image
from collectra.types.texts import Text
from collectra.usage import UsageLedger
from tests.unit.test_pipeline_stages import (  # noqa: F401  (files is a fixture)
    EVENTS,
    Describe,
    build,
    files,
)


def spend(monkeypatch, seen=None):
    """Make Describe record a priced LLM call for each page, as an LLM task does."""

    def run(self, image: Image) -> Text:
        ledger = self.context.usage_ledger
        if seen is not None:
            seen.append(ledger)
        ledger.add(
            self.name,
            "claude-haiku-5-5",
            RunUsage(input_tokens=100, output_tokens=10, cost=Decimal("0.001")),
        )
        stem = Path(image.get_path()).stem
        return Text(name=self.get_output_name(), data=f"page {stem}")

    monkeypatch.setattr(Describe, "run", run)


def test_every_file_in_a_run_shares_one_ledger(tmp_path, files, monkeypatch):
    seen = []
    spend(monkeypatch, seen)
    pipeline = build(tmp_path, batch_size=1)
    pipeline.run(files=files)
    assert len(seen) == 3 and len({id(ledger) for ledger in seen}) == 1
    assert seen[0] is pipeline.usage
    assert pipeline.usage.tasks["first"].calls == 3
    assert pipeline.usage.tasks["first"].cost == Decimal("0.003")


def test_each_run_starts_a_new_ledger(tmp_path, files, monkeypatch):
    spend(monkeypatch)
    pipeline = build(tmp_path, batch_size=1)
    pipeline.run(files=files)
    pipeline.run(files=files, force=True)
    assert pipeline.usage.tasks["first"].calls == 3


def test_a_pipeline_that_has_not_run_has_an_empty_ledger(tmp_path):
    assert isinstance(build(tmp_path).usage, UsageLedger)
    assert not build(tmp_path).usage


def test_the_terminal_run_prints_the_usage_summary(
    tmp_path, files, monkeypatch, capsys
):
    spend(monkeypatch)
    build(tmp_path, batch_size=1).cli_run([str(path) for path in files])
    out = capsys.readouterr().out
    assert "LLM usage" in out
    assert "claude-haiku-5-5" in out
    assert "$0.0030" in out


def test_a_run_without_llm_calls_prints_no_summary(tmp_path, files, capsys):
    build(tmp_path, batch_size=1).cli_run([str(path) for path in files])
    assert "LLM usage" not in capsys.readouterr().out
    EVENTS.clear()
