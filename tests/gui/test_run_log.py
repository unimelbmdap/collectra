"""Tests for the append-only run log of the GUI Run button."""

from datetime import datetime

import pytest

from collectra.gui.run_log import LOG_NAME, begin_entry, clean_note, finish_entry
from collectra.pipelines.run_plan import RunPlan, RunReport, TaskOutcome, TaskPlan

NOW = datetime(2026, 11, 10, 14, 30, 5)
PLAN = RunPlan(
    [
        TaskPlan("ocr", "run", "no output yet"),
        TaskPlan("translate", "rerun", "upstream ocr runs"),
        TaskPlan("tags", "skip", "already has output"),
    ]
)


def begin(folder, **overrides):
    args = dict(pipeline="books", version="1.2", note="", plan=PLAN, now=NOW)
    args.update(overrides)
    begin_entry(folder, **args)


def test_log_name():
    assert LOG_NAME == "run-log.txt"


def test_first_entry_creates_file(tmp_path):
    begin(tmp_path, note="first go")
    text = (tmp_path / LOG_NAME).read_text(encoding="utf-8")
    assert "2026-11-10 14:30:05" in text
    assert "books" in text and "1.2" in text
    assert "first go" in text
    assert "ocr" in text and "translate" in text and "tags" in text
    assert "no output yet" in text


def test_entry_without_version_or_note(tmp_path):
    begin(tmp_path, version=None)
    text = (tmp_path / LOG_NAME).read_text(encoding="utf-8")
    assert "books" in text
    assert "None" not in text
    assert "note" not in text.lower()


def test_default_now_is_used(tmp_path):
    begin_entry(tmp_path, pipeline="p", version=None, note="", plan=PLAN)
    assert str(datetime.now().year) in (tmp_path / LOG_NAME).read_text("utf-8")


def test_two_runs_leave_two_entries_first_intact(tmp_path):
    begin(tmp_path)
    first = (tmp_path / LOG_NAME).read_bytes()
    begin(tmp_path, note="second")
    both = (tmp_path / LOG_NAME).read_bytes()
    assert both.startswith(first)
    assert len(both) > len(first)
    assert both.decode("utf-8").count("2026-11-10 14:30:05") == 2


def test_hand_edits_survive_later_append(tmp_path):
    begin(tmp_path)
    with open(tmp_path / LOG_NAME, "a", encoding="utf-8") as handle:
        handle.write("my own remark\n")
    before = (tmp_path / LOG_NAME).read_bytes()
    finish_entry(tmp_path, RunReport([TaskOutcome("ocr", "ran", "")]), 1.0)
    after = (tmp_path / LOG_NAME).read_bytes()
    assert after.startswith(before)
    assert b"my own remark" in after


def test_clean_note_collapses_whitespace():
    assert clean_note("a\nb\r\n  c") == "a b c"
    assert clean_note("  padded \t") == "padded"


def test_clean_note_caps_length():
    assert len(clean_note("x" * 500)) == 200


def test_non_ascii_note_round_trips(tmp_path):
    begin(tmp_path, note="café 写真 ✓")
    assert "café 写真 ✓" in (tmp_path / LOG_NAME).read_text(encoding="utf-8")


def test_unwritable_folder_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        begin(tmp_path / "missing" / "deeper")


def test_finish_entry_lines_and_summary(tmp_path):
    report = RunReport(
        [
            TaskOutcome("a", "ran", ""),
            TaskOutcome("b", "re-ran", "upstream a ran"),
            TaskOutcome("c", "skipped", "already has output"),
            TaskOutcome("d", "failed", "boom\nsecond line"),
            TaskOutcome("e", "cancelled", ""),
        ]
    )
    finish_entry(tmp_path, report, 12.34)
    lines = (tmp_path / LOG_NAME).read_text(encoding="utf-8").splitlines()
    assert any(line.strip() == "a  ran" for line in lines)
    assert any(
        line.strip() == "b  re-ran, replaced existing output (upstream a ran)"
        for line in lines
    )
    assert any(line.strip() == "c  skipped (kept existing output)" for line in lines)
    assert any(line.strip() == "d  failed: boom second line" for line in lines)
    assert any(line.strip() == "e  cancelled" for line in lines)
    assert (
        lines[-1].strip() == "result  2 ran, 1 skipped, 1 failed, 1 cancelled, 12.3 s"
    )


def test_summary_omits_cancelled_and_counts_rerun_as_ran(tmp_path):
    report = RunReport(
        [TaskOutcome("a", "ran", ""), TaskOutcome("b", "re-ran", "upstream a ran")]
    )
    finish_entry(tmp_path, report, 0.04)
    last = (tmp_path / LOG_NAME).read_text(encoding="utf-8").splitlines()[-1]
    assert last.strip() == "result  2 ran, 0 skipped, 0 failed, 0.0 s"


def test_rerun_without_upstream_in_detail(tmp_path):
    finish_entry(tmp_path, RunReport([TaskOutcome("b", "re-ran", "odd")]), 1)
    text = (tmp_path / LOG_NAME).read_text(encoding="utf-8")
    assert "b  re-ran, replaced existing output\n" in text


def test_cancelled_detail_is_kept(tmp_path):
    detail = "kept previous output, which may be out of date (upstream a ran)"
    finish_entry(tmp_path, RunReport([TaskOutcome("b", "cancelled", detail)]), 1)
    text = (tmp_path / LOG_NAME).read_text(encoding="utf-8")
    assert f"b  cancelled: {detail}\n" in text


def test_finish_entry_unwritable_raises(tmp_path):
    with pytest.raises(OSError):
        finish_entry(tmp_path / "nope", RunReport([]), 1)


def test_rerun_that_produced_nothing_says_it_kept_the_output(tmp_path):
    detail = "produced nothing, kept existing output (upstream a ran)"
    finish_entry(tmp_path, RunReport([TaskOutcome("b", "re-ran", detail)]), 1)
    text = (tmp_path / LOG_NAME).read_text(encoding="utf-8")
    assert "b  re-ran, produced nothing, kept existing output (upstream a ran)\n" in (
        text
    )
    assert "replaced" not in text


def test_plan_warnings_go_in_the_started_block(tmp_path):
    plan = RunPlan(
        PLAN.tasks,
        warnings=("pipeline version 1.4.2, results made with 1.3.0",),
    )
    begin(tmp_path, plan=plan, note="mine")
    lines = (tmp_path / LOG_NAME).read_text(encoding="utf-8").splitlines()
    assert "  note: mine" in lines
    assert "  warning: pipeline version 1.4.2, results made with 1.3.0" in lines
    assert lines.index("  warning: pipeline version 1.4.2, results made with 1.3.0") < (
        lines.index("  ocr  planned to run: no output yet")
    )


def test_no_warning_line_without_warnings(tmp_path):
    begin(tmp_path)
    assert "warning" not in (tmp_path / LOG_NAME).read_text(encoding="utf-8")
