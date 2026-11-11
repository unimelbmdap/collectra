"""Running several pages in one pass, each with its own report."""

import shutil
from pathlib import Path

import pytest
import yaml

from collectra.pipelines.run_plan import TaskPlan
from tests.unit import test_pipeline_stages as stages
from tests.unit.test_run_page import (
    RAN,
    Classify,
    Detect,
    Ocr,
    build,
    drop,
    outcome,
    saved,
    statuses,
)

TASKS = {"detect", "ocr", "classify"}


@pytest.fixture
def pages(tmp_path, raw_img_path):
    """Two pages that every task has already run on once."""
    RAN.clear()
    source = tmp_path / "input"
    source.mkdir()
    for name in "ab":
        shutil.copy(raw_img_path, source / f"{name}.jpg")
    build(tmp_path).run(files=[str(source / "a.jpg"), str(source / "b.jpg")])
    RAN.clear()
    return source / "a.pge", source / "b.pge"


def run_pages(tmp_path, folders, rerun=frozenset(), overwrite=False, **kwargs):
    pipeline = build(tmp_path)
    plans = {folder: pipeline.plan_page_run(folder, overwrite) for folder in folders}
    return pipeline.run_pages(plans, set(rerun), **kwargs)


def outputs(folder: Path) -> dict:
    """Each task output's results entry and the bytes of its file."""
    data = yaml.safe_load((folder / "results.yaml").read_text())
    return {
        name: (data[name], (folder / data[name]["path"]).read_bytes())
        for name in ("boxes", "words", "label")
        if name in data
    }


def on_page(name, task):
    """Make ``task`` raise on the page named ``name`` only."""
    real = task.run

    def run(self, item):
        if self.context.file_path.name == name:
            raise RuntimeError(f"cannot read {name}")
        return real(self, item)

    return run


def test_two_pages_get_their_own_reports(tmp_path, pages):
    a, b = pages
    drop(a, "label")
    reports = run_pages(tmp_path, pages)
    assert RAN == ["classify"]
    assert statuses(reports[a]) == {
        "detect": "skipped",
        "ocr": "skipped",
        "classify": "ran",
    }
    assert set(statuses(reports[b]).values()) == {"skipped"}


def test_default_run_replaces_nothing(tmp_path, pages):
    a, b = pages
    drop(a, "words")
    before = {a: outputs(a), b: outputs(b)}
    reports = run_pages(tmp_path, pages)
    assert RAN == ["ocr"]
    after_a = outputs(a)
    assert {name: after_a[name] for name in before[a]} == before[a]
    assert outputs(b) == before[b]
    assert outcome(reports[a], "classify").status == "skipped"


def test_overwrite_reruns_every_task(tmp_path, pages, monkeypatch):
    a, b = pages
    boxes = {a: saved(a, "boxes"), b: saved(b, "boxes")}
    words = {a: saved(a, "words"), b: saved(b, "words")}

    def broken(self, image):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(Detect, "run", broken)
    reports = run_pages(tmp_path, pages, rerun=TASKS, overwrite=True)
    assert sorted(RAN) == ["classify", "classify", "ocr", "ocr"]
    for folder in pages:
        assert statuses(reports[folder]) == {
            "detect": "failed",
            "ocr": "re-ran",
            "classify": "re-ran",
        }
        assert saved(folder, "boxes") == boxes[folder]
        assert saved(folder, "words") != words[folder]


def test_page_done_is_called_once_per_page(tmp_path, pages):
    a, b = pages
    drop(a, "label")
    drop(b, "words")
    done = []
    run_pages(tmp_path, pages, page_done=done.append)
    assert done == [a, b]


def test_page_done_waits_for_a_page_held_at_a_batched_task(tmp_path, raw_img_path):
    stages.EVENTS.clear()
    for name in "abc":
        shutil.copy(raw_img_path, tmp_path / f"{name}.jpg")
    files = [str(tmp_path / f"{name}.jpg") for name in "abc"]

    def done(file):
        stages.EVENTS.append(f"done:{Path(file).stem}")

    stages.build(tmp_path).run(files=files, page_done=done)
    assert stages.EVENTS == [
        "describe:a", "describe:b", "prefetch:2",
        "shout:a", "done:a", "shout:b", "done:b", "clear",
        "describe:c", "prefetch:1", "shout:c", "done:c", "clear",
    ]  # fmt: skip


def test_stop_between_tasks_marks_the_rest_cancelled_on_every_page(tmp_path, pages):
    a, b = pages
    for folder in pages:
        drop(folder, "boxes", "words", "label")
    reports = run_pages(tmp_path, pages, should_stop=lambda: len(RAN) >= 1)
    assert len(RAN) == 1
    assert outcome(reports[a], RAN[0]).status == "ran"
    rest = [
        item
        for folder in pages
        for item in reports[folder].outcomes
        if (folder, item.name) != (a, RAN[0])
    ]
    assert len(rest) == 5
    assert {(item.status, item.detail) for item in rest} == {
        ("cancelled", "stopped before it started")
    }


def test_one_failing_page_does_not_stop_the_other(tmp_path, pages, monkeypatch):
    a, b = pages
    for folder in pages:
        drop(folder, "words", "label")
    monkeypatch.setattr(Ocr, "run", on_page("a.pge", Ocr))
    reports = run_pages(tmp_path, pages)
    assert "cannot read a.pge" in outcome(reports[a], "ocr").detail
    assert statuses(reports[a]) == {
        "detect": "skipped",
        "ocr": "failed",
        "classify": "cancelled",
    }
    assert statuses(reports[b]) == {
        "detect": "skipped",
        "ocr": "ran",
        "classify": "ran",
    }
    assert saved(b, "label").startswith("label of words")


def test_a_page_that_cannot_be_read_fails_only_its_own_tasks(
    tmp_path, pages, monkeypatch
):
    a, b = pages
    pipeline = build(tmp_path)
    for folder in pages:
        drop(folder, "words", "label")
    plans = {folder: pipeline.plan_page_run(folder) for folder in pages}
    (a.parent / "a.jpg").unlink()
    (a / "a.jpg").write_bytes(b"not an image")
    monkeypatch.setattr(Ocr, "run", on_page("b.pge", Ocr))
    reports = pipeline.run_pages(plans, set())
    assert {item.status for item in reports[a].outcomes} == {"failed"}
    assert all("a.jpg could not be read" in i.detail for i in reports[a].outcomes)
    assert outcome(reports[b], "classify").status == "cancelled"
    assert outcome(reports[b], "classify").detail == (
        "did not run: its inputs were not produced"
    )


def test_plan_with_overwrite(tmp_path, pages):
    a, _ = pages
    drop(a, "label")
    assert build(tmp_path).plan_page_run(a, overwrite=True).tasks == [
        TaskPlan("detect", "rerun", "overwrite"),
        TaskPlan("ocr", "rerun", "overwrite"),
        TaskPlan("classify", "run", "no output yet"),
    ]


def test_overwrite_report_has_no_upstream_clause(tmp_path, pages, monkeypatch):
    a, _ = pages
    pipeline = build(tmp_path)
    report = pipeline.run_page(a, pipeline.plan_page_run(a, overwrite=True))
    assert {(item.status, item.detail) for item in report.outcomes} == {
        ("re-ran", "replaced existing output")
    }

    def broken(self, text):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(Classify, "run", broken)
    report = pipeline.run_page(a, pipeline.plan_page_run(a, overwrite=True))
    assert outcome(report, "classify").detail == (
        "RuntimeError: out of memory; kept previous output"
    )

    RAN.clear()
    plan = pipeline.plan_page_run(a, overwrite=True)
    report = pipeline.run_page(a, plan, lambda: len(RAN) >= 1)
    stopped = [item for item in report.outcomes if item.status == "cancelled"]
    assert len(stopped) == 2
    assert {item.detail for item in stopped} == {
        "stopped before it started; kept previous output"
    }
