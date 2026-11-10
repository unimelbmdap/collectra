"""Which tasks a page run runs, re-runs or skips."""

import shutil
from pathlib import Path

import pytest
import yaml

from collectra import Pipeline
from collectra.pipelines.run_plan import RunPlan, TaskPlan, plan_tasks
from collectra.tasks.base import Task
from collectra.types.images import Image
from collectra.types.texts import Text

THIS = "tests.unit.test_run_plan"
RAN: list[str] = []

CHAIN = {"ocr": {"classify"}, "classify": set(), "detect": set()}
ORDER = ["detect", "ocr", "classify"]


def by_name(plan: RunPlan) -> dict[str, TaskPlan]:
    return {task.name: task for task in plan.tasks}


def test_missing_task_runs():
    plan = by_name(plan_tasks({"ocr"}, CHAIN, ORDER))
    assert plan["ocr"] == TaskPlan("ocr", "run", "no output yet")
    assert plan["classify"] == TaskPlan("classify", "rerun", "upstream ocr runs")
    assert plan["detect"] == TaskPlan("detect", "skip", "already has output")


def test_order_follows_the_given_order():
    plan = plan_tasks({"ocr"}, CHAIN, ORDER)
    assert [task.name for task in plan.tasks] == ORDER
    assert plan.to_run == ["ocr", "classify"]


def test_downstream_is_transitive():
    children = {"a": {"b"}, "b": {"c"}, "c": set()}
    plan = by_name(plan_tasks({"a"}, children, ["a", "b", "c"]))
    assert plan["b"] == TaskPlan("b", "rerun", "upstream a runs")
    assert plan["c"].action == "rerun"


def test_reason_names_the_nearest_running_upstream_task():
    children = {"a": {"b"}, "b": {"c"}, "c": set()}
    plan = by_name(plan_tasks({"a", "b"}, children, ["a", "b", "c"]))
    assert plan["c"].reason == "upstream b runs"


def test_nothing_missing_skips_everything():
    plan = plan_tasks(set(), CHAIN, ORDER)
    assert plan.to_run == []
    assert {task.action for task in plan.tasks} == {"skip"}


def test_all_missing_runs_everything_and_nothing_is_rerun():
    plan = plan_tasks(set(ORDER), CHAIN, ORDER)
    assert {task.action for task in plan.tasks} == {"run"}
    assert plan.to_run == ORDER


def test_downstream_that_is_also_missing_is_run_not_rerun():
    plan = by_name(plan_tasks({"ocr", "classify"}, CHAIN, ORDER))
    assert plan["classify"] == TaskPlan("classify", "run", "no output yet")


class Describe(Task):
    def run(self, image: Image) -> Text:
        RAN.append("describe")
        return Text(name=self.get_output_name(), data="page")


class Shout(Task):
    def run(self, text: Text) -> Text:
        RAN.append("shout")
        return Text(name=self.get_output_name(), data=text().upper())


def build(tmp_path):
    return Pipeline(
        "Plan",
        "pln",
        "0.1.0",
        path=str(tmp_path),
        describe={
            "type": f"{THIS}.Describe",
            "input": ["page"],
            "output": ["described"],
        },
        shout={"type": f"{THIS}.Shout", "input": ["described"], "output": ["shouted"]},
    )


@pytest.fixture
def folder(tmp_path, raw_img_path):
    RAN.clear()
    source = tmp_path / "input"
    source.mkdir()
    shutil.copy(raw_img_path, source / "a.jpg")
    build(tmp_path).run(files=[str(source / "a.jpg")])
    RAN.clear()
    return source / "a.pln"


def edit_results(folder: Path, change):
    path = folder / "results.yaml"
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data))


def test_page_with_all_output_skips_everything(tmp_path, folder):
    plan = build(tmp_path).plan_page_run(folder)
    assert plan.to_run == []
    assert [task.name for task in plan.tasks] == ["describe", "shout"]
    assert RAN == []


def test_page_missing_the_first_output_runs_it_and_reruns_the_next(tmp_path, folder):
    edit_results(folder, lambda data: data.pop("described"))
    plan = by_name(build(tmp_path).plan_page_run(folder))
    assert plan["describe"].action == "run"
    assert plan["shout"] == TaskPlan("shout", "rerun", "upstream describe runs")
    assert RAN == []


def test_page_missing_only_the_last_output_runs_just_that(tmp_path, folder):
    edit_results(folder, lambda data: data.pop("shouted"))
    plan = by_name(build(tmp_path).plan_page_run(folder))
    assert plan["describe"].action == "skip"
    assert plan["shout"].action == "run"


def test_a_missing_image_copy_is_reported(tmp_path, folder):
    (folder / "a.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="a.jpg"):
        build(tmp_path).plan_page_run(folder)


def test_a_folder_without_results_is_reported(tmp_path):
    (tmp_path / "empty.pln").mkdir()
    with pytest.raises(FileNotFoundError, match="results.yaml"):
        build(tmp_path).plan_page_run(tmp_path / "empty.pln")


def test_results_that_do_not_match_the_pipeline_are_reported(tmp_path, folder):
    edit_results(folder, lambda data: data.pop("page"))
    with pytest.raises(ValueError, match="do not match"):
        build(tmp_path).plan_page_run(folder)
