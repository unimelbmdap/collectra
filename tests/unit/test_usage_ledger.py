"""The run's LLM usage: tokens and cost per task, and the summary table."""

import threading
from decimal import Decimal

from pydantic_ai.usage import RunUsage
from rich.console import Console

from collectra.usage import UsageLedger


def usage(
    input_tokens=100,
    output_tokens=10,
    cache_read=0,
    cache_write=0,
    requests=1,
    cost=Decimal("0.01"),
):
    return RunUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        requests=requests,
        cost=cost,
    )


def printed(ledger: UsageLedger) -> str:
    console = Console(width=200, record=True, color_system=None)
    ledger.print_summary(console)
    return console.export_text()


def test_a_new_ledger_is_empty():
    ledger = UsageLedger()
    assert not ledger
    assert ledger.tasks == {}


def test_usage_adds_up_by_task():
    ledger = UsageLedger()
    ledger.add("reader", "claude-opus-5-5", usage(cache_read=40, cache_write=20))
    ledger.add("reader", "claude-opus-5-5", usage(requests=2, cost=Decimal("0.02")))
    ledger.add("tidy", "claude-haiku-5-5", usage(input_tokens=5, output_tokens=1))
    assert ledger
    reader = ledger.tasks["reader"]
    assert reader.model == "claude-opus-5-5"
    assert reader.calls == 2
    assert reader.requests == 3
    assert reader.input_tokens == 200
    assert reader.output_tokens == 20
    assert reader.cache_read_tokens == 40
    assert reader.cache_write_tokens == 20
    assert reader.cost == Decimal("0.03")
    assert reader.unpriced == 0
    assert list(ledger.tasks) == ["reader", "tidy"]


def test_calls_that_could_not_be_priced_are_counted():
    ledger = UsageLedger()
    ledger.add("local", "ollama:llama3", usage(cost=None))
    ledger.add("local", "ollama:llama3", usage(cost=None))
    assert ledger.tasks["local"].cost == 0
    assert ledger.tasks["local"].unpriced == 2


def test_the_ledger_can_be_added_to_from_several_threads():
    ledger = UsageLedger()

    def add_many():
        for _ in range(500):
            ledger.add("reader", "m", usage())

    threads = [threading.Thread(target=add_many) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert ledger.tasks["reader"].calls == 2000
    assert ledger.tasks["reader"].cost == Decimal("20.00")


def test_the_summary_lists_each_task_and_a_total():
    ledger = UsageLedger()
    ledger.add(
        "reader",
        "claude-opus-5-5",
        usage(
            input_tokens=12_345,
            output_tokens=678,
            cache_read=4_000,
            cache_write=3_000,
            cost=Decimal("0.0912"),
        ),
    )
    ledger.add(
        "tidy",
        "claude-haiku-5-5",
        usage(input_tokens=1_000, output_tokens=50, cost=Decimal("0.0006")),
    )
    text = printed(ledger)
    assert "LLM usage" in text
    for heading in ("Task", "Model", "Calls", "Input", "Cache", "read", "write"):
        assert heading in text
    assert "Output" in text and "Cost" in text
    assert "reader" in text and "claude-opus-5-5" in text
    assert "12,345" in text and "4,000" in text and "3,000" in text
    assert "$0.0912" in text
    assert "$0.0006" in text
    total = next(line for line in text.splitlines() if "Total" in line)
    assert "13,345" in total and "728" in total and "$0.0918" in total


def test_the_summary_fits_an_80_column_terminal():
    ledger = UsageLedger()
    ledger.add(
        "line_reader",
        "claude-opus-5-5",
        usage(1_234_567, 34_567, 800_000, 90_000, cost=Decimal("3.41")),
    )
    console = Console(width=80, record=True, color_system=None)
    ledger.print_summary(console)
    text = console.export_text()
    assert "line_reader" in text and "claude-opus-5-5" in text
    assert "1,234,567" in text and "800,000" in text and "$3.41" in text
    assert "…" not in text


def test_costs_of_a_dollar_or_more_show_cents():
    ledger = UsageLedger()
    ledger.add("reader", "m", usage(cost=Decimal("12.3456")))
    assert "$12.35" in printed(ledger)


def test_a_task_that_could_not_be_priced_says_so():
    ledger = UsageLedger()
    ledger.add("local", "ollama:llama3", usage(cost=None))
    text = printed(ledger)
    local = next(line for line in text.splitlines() if "local" in line)
    assert "unknown" in local
    total = next(line for line in text.splitlines() if "Total" in line)
    assert "unknown" in total
    assert "could not be priced" in text


def test_a_partly_priced_cost_is_shown_as_a_lower_bound():
    ledger = UsageLedger()
    ledger.add("mixed", "m", usage(cost=Decimal("0.05")))
    ledger.add("mixed", "m", usage(cost=None))
    line = next(line for line in printed(ledger).splitlines() if "mixed" in line)
    assert "≥ $0.0500" in line


def test_the_summary_explains_what_input_includes():
    ledger = UsageLedger()
    ledger.add("reader", "m", usage())
    assert "Input includes the cached tokens" in printed(ledger)


def test_names_that_look_like_markup_are_printed_as_typed():
    ledger = UsageLedger()
    ledger.add("[bold]reader", "[red]m", usage())
    text = printed(ledger)
    assert "[bold]reader" in text and "[red]m" in text


def test_an_empty_ledger_prints_nothing():
    assert printed(UsageLedger()).strip() == ""
