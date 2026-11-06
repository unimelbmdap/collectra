"""A task that fails stops only its own branch of the DAG; the rest keeps going."""

import shutil
from pathlib import Path

import pytest
import yaml

from collectra import Pipeline
from collectra.tasks.base import Task
from collectra.types.images import Image
from collectra.types.texts import Text

THIS = "tests.unit.test_pipeline_failures"
RAN: list[str] = []


class Boom(Task):
    def run(self, image: Image) -> Text:
        RAN.append(f"boom:{Path(image.get_path()).stem}")
        raise RuntimeError("model is not installed")


class Describe(Task):
    def run(self, image: Image) -> Text:
        RAN.append(f"describe:{Path(image.get_path()).stem}")
        return Text(name=self.get_output_name(), data="page")


class Shout(Task):
    def run(self, text: Text) -> Text:
        RAN.append("shout")
        return Text(name=self.get_output_name(), data=text().upper())


class Join(Task):
    def run(self, first: Text, second: Text) -> Text:
        RAN.append("join")
        return Text(name=self.get_output_name(), data=first() + second())


def build(tmp_path):
    return Pipeline(
        "Failing",
        "fls",
        "0.1.0",
        path=str(tmp_path),
        boom={"type": f"{THIS}.Boom", "input": ["page"], "output": ["broken"]},
        after_boom={"type": f"{THIS}.Shout", "input": ["broken"], "output": ["lost"]},
        describe={
            "type": f"{THIS}.Describe",
            "input": ["page"],
            "output": ["described"],
        },
        after_describe={
            "type": f"{THIS}.Shout",
            "input": ["described"],
            "output": ["shouted"],
        },
        join={
            "type": f"{THIS}.Join",
            "input": ["broken", "shouted"],
            "output": ["joined"],
        },
    )


@pytest.fixture
def files(tmp_path, raw_img_path):
    RAN.clear()
    directory = tmp_path / "input"
    directory.mkdir()
    paths = []
    for name in "ab":
        path = directory / f"{name}.jpg"
        shutil.copy(raw_img_path, path)
        paths.append(str(path))
    return paths


def results(tmp_path, name="a"):
    return yaml.safe_load(
        (tmp_path / "input" / f"{name}.fls" / "results.yaml").read_text()
    )


def test_the_other_branch_finishes_when_a_task_fails(tmp_path, files):
    build(tmp_path).run(files=files[:1])
    data = results(tmp_path)
    assert data["described"]["data"] == "page"
    assert data["shouted"]["data"] == "PAGE"


def test_tasks_that_need_the_failed_output_are_not_run(tmp_path, files):
    build(tmp_path).run(files=files[:1])
    data = results(tmp_path)
    assert "lost" not in data and "joined" not in data
    assert "join" not in RAN
    assert RAN.count("shout") == 1  # only the one after the working branch


def test_every_file_is_still_processed(tmp_path, files):
    build(tmp_path).run(files=files)
    assert sorted(item for item in RAN if item.startswith("boom")) == [
        "boom:a",
        "boom:b",
    ]
    assert results(tmp_path, "b")["shouted"]["data"] == "PAGE"


def test_the_failure_is_logged_with_its_traceback_and_summarised(
    tmp_path, files, caplog
):
    with caplog.at_level("ERROR", logger="collectra"):
        build(tmp_path).run(files=files)
    failures = [r for r in caplog.records if "boom" in r.getMessage()]
    assert any(r.exc_info for r in failures)
    assert any("model is not installed" in r.getMessage() for r in failures)
    summary = caplog.records[-1].getMessage()
    assert "2 task failure(s)" in summary
    assert "boom" in summary and "a.fls" in summary and "b.fls" in summary


def test_no_summary_when_nothing_fails(tmp_path, files, caplog):
    pipeline = Pipeline(
        "Failing",
        "fls",
        "0.1.0",
        path=str(tmp_path),
        describe={
            "type": f"{THIS}.Describe",
            "input": ["page"],
            "output": ["described"],
        },
    )
    with caplog.at_level("ERROR", logger="collectra"):
        pipeline.run(files=files[:1])
    assert caplog.records == []


def test_a_keyboard_interrupt_still_stops_the_run(tmp_path, files, monkeypatch):
    def interrupted(self, image):
        raise KeyboardInterrupt

    monkeypatch.setattr(Boom, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        build(tmp_path).run(files=files[:1])
