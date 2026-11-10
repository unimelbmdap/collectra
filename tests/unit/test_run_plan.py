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


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file under a folder, by relative path, with its bytes."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_planning_writes_nothing(tmp_path, folder):
    edit_results(folder, lambda data: data.pop("shouted"))
    before = snapshot(folder.parent)
    build(tmp_path).plan_page_run(folder)
    assert snapshot(folder.parent) == before


def test_a_folder_not_named_for_this_pipeline_is_refused_untouched(tmp_path, folder):
    renamed = folder.with_name("a_renamed")
    folder.rename(renamed)
    before = snapshot(renamed)
    assert set(yaml.safe_load(before["results.yaml"])) >= {"page", "described"}
    with pytest.raises(ValueError, match=r"a_renamed.*\.pln"):
        build(tmp_path).plan_page_run(renamed)
    assert snapshot(renamed) == before
    assert RAN == []


def test_a_page_of_another_pipeline_is_refused_untouched(tmp_path, folder):
    other = folder.with_suffix(".other")
    folder.rename(other)
    before = snapshot(other)
    with pytest.raises(ValueError, match="cannot be run with the Plan pipeline"):
        build(tmp_path).plan_page_run(other)
    assert snapshot(other) == before


def test_a_matching_page_has_no_warnings(tmp_path, folder):
    assert build(tmp_path).plan_page_run(folder).warnings == ()


def test_a_plan_made_without_warnings_has_none():
    assert RunPlan([]).warnings == ()


def test_a_different_pipeline_version_is_warned_about_not_blocked(tmp_path, folder):
    pipeline = build(tmp_path)
    pipeline.version = "0.2.0"
    plan = pipeline.plan_page_run(folder)
    assert plan.warnings == ("pipeline version 0.2.0, results made with 0.1.0",)
    assert [task.name for task in plan.tasks] == ["describe", "shout"]


def test_results_made_by_another_workflow_are_warned_about(tmp_path, folder):
    def rename(data):
        data["results"]["workflow"] = "Other"

    edit_results(folder, rename)
    plan = build(tmp_path).plan_page_run(folder)
    assert plan.warnings == ("these results were made by the Other pipeline, not Plan",)


def test_results_without_a_header_have_no_warnings(tmp_path, folder):
    edit_results(folder, lambda data: data.pop("results"))
    assert build(tmp_path).plan_page_run(folder).warnings == ()


def test_a_page_image_that_cannot_be_read_is_reported(tmp_path, folder):
    (folder.parent / "a.jpg").unlink()
    (folder / "a.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="The page image a.jpg could not be read"):
        build(tmp_path).plan_page_run(folder)
