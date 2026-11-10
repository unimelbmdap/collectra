"""Running one page's planned tasks, with stop and failure handling."""

import shutil
from pathlib import Path

import pytest
import yaml

from collectra import Pipeline
from collectra.pipelines.run_plan import RunReport, TaskOutcome
from collectra.tasks.base import Task
from collectra.types.images import Image
from collectra.types.texts import Text

THIS = "tests.unit.test_run_page"
RAN: list[str] = []


def saved_text(task: Task, data: str) -> Text:
    """A text written to its own file in the results folder, like real tasks do."""
    text = Text(name=task.get_output_name(), data=data)
    text._file_extension = "txt"
    return text


class Detect(Task):
    def run(self, image: Image) -> Text:
        RAN.append("detect")
        return saved_text(self, f"boxes {len(RAN)}")


class Ocr(Task):
    def run(self, image: Image) -> Text:
        RAN.append("ocr")
        return saved_text(self, f"words {len(RAN)}")


class Classify(Task):
    def run(self, text: Text) -> Text:
        RAN.append("classify")
        return saved_text(self, f"label of {text()} {len(RAN)}")


def build(tmp_path):
    return Pipeline(
        "Page",
        "pge",
        "0.1.0",
        path=str(tmp_path),
        detect={"type": f"{THIS}.Detect", "input": ["page"], "output": ["boxes"]},
        ocr={"type": f"{THIS}.Ocr", "input": ["page"], "output": ["words"]},
        classify={
            "type": f"{THIS}.Classify",
            "input": ["words"],
            "output": ["label"],
        },
    )


@pytest.fixture
def folder(tmp_path, raw_img_path):
    """A page that every task has already run on once."""
    RAN.clear()
    source = tmp_path / "input"
    source.mkdir()
    shutil.copy(raw_img_path, source / "a.jpg")
    build(tmp_path).run(files=[str(source / "a.jpg")])
    RAN.clear()
    return source / "a.pge"


def drop(folder: Path, *names: str) -> None:
    """Remove outputs from a page, as if those tasks had never run."""
    path = folder / "results.yaml"
    data = yaml.safe_load(path.read_text())
    for name in names:
        entry = data.pop(name)
        (folder / entry["path"]).unlink()
    path.write_text(yaml.safe_dump(data))


def saved(folder: Path, name: str) -> str:
    data = yaml.safe_load((folder / "results.yaml").read_text())
    return (folder / data[name]["path"]).read_text()


def statuses(report: RunReport) -> dict[str, str]:
    return {outcome.name: outcome.status for outcome in report.outcomes}


def outcome(report: RunReport, name: str) -> TaskOutcome:
    return next(item for item in report.outcomes if item.name == name)


def run_page(tmp_path, folder, should_stop=None):
    pipeline = build(tmp_path)
    plan = pipeline.plan_page_run(folder)
    if should_stop is None:
        return pipeline.run_page(folder, plan)
    return pipeline.run_page(folder, plan, should_stop)


def test_skips_tasks_with_output_and_runs_missing(tmp_path, folder):
    boxes = (folder / "boxes" / "boxes.txt").read_bytes()
    words = (folder / "words" / "words.txt").read_bytes()
    drop(folder, "label")
    report = run_page(tmp_path, folder)
    assert RAN == ["classify"]
    assert (folder / "boxes" / "boxes.txt").read_bytes() == boxes
    assert (folder / "words" / "words.txt").read_bytes() == words
    assert saved(folder, "label").startswith("label of words")
    assert statuses(report) == {
        "detect": "skipped",
        "ocr": "skipped",
        "classify": "ran",
    }
    assert [item.name for item in report.outcomes] == ["detect", "ocr", "classify"]
    assert report.failed == []


def test_downstream_reruns_and_replaces_output(tmp_path, folder):
    old = saved(folder, "label")
    drop(folder, "words")
    report = run_page(tmp_path, folder)
    assert RAN == ["ocr", "classify"]
    assert saved(folder, "label") != old
    assert saved(folder, "label") == "label of words 1 2"
    rerun = outcome(report, "classify")
    assert rerun.status == "re-ran"
    assert "upstream ocr ran" in rerun.detail
    assert "replaced existing output" in rerun.detail
    assert outcome(report, "detect").status == "skipped"


def test_all_missing_runs_everything(tmp_path, folder):
    drop(folder, "boxes", "words", "label")
    report = run_page(tmp_path, folder)
    assert sorted(RAN) == ["classify", "detect", "ocr"]
    assert set(statuses(report).values()) == {"ran"}
    assert saved(folder, "boxes").startswith("boxes")


def test_failed_task_leaves_no_partial_output(tmp_path, folder, monkeypatch):
    drop(folder, "boxes", "words")

    def broken(self, image):
        RAN.append("ocr")
        partial = self.context.file_path / "words"
        partial.mkdir(exist_ok=True)
        (partial / "words.txt").write_text("half")
        raise RuntimeError("model is not installed")

    monkeypatch.setattr(Ocr, "run", broken)
    pipeline = build(tmp_path)
    report = pipeline.run_page(folder, pipeline.plan_page_run(folder))
    failed = outcome(report, "ocr")
    assert failed.status == "failed"
    assert "model is not installed" in failed.detail
    assert report.failed == [failed]
    assert outcome(report, "detect").status == "ran"
    assert outcome(report, "classify").status == "cancelled"
    assert "words" not in yaml.safe_load((folder / "results.yaml").read_text())
    node = pipeline.node_manager.resolve_node("ocr")
    assert not pipeline._has_output(node)

    monkeypatch.undo()
    RAN.clear()
    again = build(tmp_path)
    plan = again.plan_page_run(folder)
    assert [task.name for task in plan.tasks if task.action == "run"] == ["ocr"]
    report = again.run_page(folder, plan)
    assert outcome(report, "ocr").status == "ran"
    assert RAN == ["ocr", "classify"]
    assert saved(folder, "words") == "words 1"


def test_failed_rerun_keeps_the_output_it_was_to_replace(tmp_path, folder, monkeypatch):
    old = saved(folder, "label")
    drop(folder, "words")

    def broken(self, text):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(Classify, "run", broken)
    report = run_page(tmp_path, folder)
    assert outcome(report, "classify").status == "failed"
    assert saved(folder, "label") == old


def test_rerun_that_produces_nothing_keeps_existing_output(
    tmp_path, folder, monkeypatch
):
    old = saved(folder, "label")
    drop(folder, "words")

    def nothing(self, text: Text) -> Text:
        return None

    monkeypatch.setattr(Classify, "run", nothing)
    report = run_page(tmp_path, folder)
    rerun = outcome(report, "classify")
    assert rerun.status == "re-ran"
    assert "kept existing output" in rerun.detail
    assert "upstream ocr ran" in rerun.detail
    assert saved(folder, "label") == old


def test_stop_between_tasks(tmp_path, folder):
    drop(folder, "boxes", "words", "label")
    report = run_page(tmp_path, folder, should_stop=lambda: len(RAN) >= 1)
    assert len(RAN) == 1
    first = RAN[0]
    assert outcome(report, first).status == "ran"
    others = [item for item in report.outcomes if item.name != first]
    assert {item.status for item in others} == {"cancelled"}
    assert saved(folder, "boxes" if first == "detect" else "words")


def test_a_missing_image_copy_is_reported(tmp_path, folder):
    pipeline = build(tmp_path)
    plan = pipeline.plan_page_run(folder)
    (folder / "a.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="a.jpg"):
        pipeline.run_page(folder, plan)
    assert RAN == []


def test_cli_run_unchanged(tmp_path, folder, caplog):
    with caplog.at_level("WARNING", logger="collectra"):
        build(tmp_path).run(files=[str(folder)])
    assert RAN == []
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "Skipping task 'ocr'" in m and "Use --force to rerun it" in m for m in messages
    )


def test_run_options_default_to_no_rerun_and_no_stop(tmp_path):
    options = build(tmp_path)._extract_run_options({"files": []})
    assert options["rerun"] == set()
    assert options["stop"]() is False
    assert options["task_overrides"] == {}


def test_rerun_and_stop_are_not_task_overrides(tmp_path):
    options = build(tmp_path)._extract_run_options(
        {"rerun": {"ocr"}, "stop": lambda: True, "temperature": 0}
    )
    assert options["rerun"] == {"ocr"}
    assert options["task_overrides"] == {"temperature": 0}


def test_report_lists_only_failures_as_failed():
    report = RunReport(
        [TaskOutcome("a", "ran", ""), TaskOutcome("b", "failed", "Boom: x")]
    )
    assert report.failed == [TaskOutcome("b", "failed", "Boom: x")]
