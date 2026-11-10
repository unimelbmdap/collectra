"""Tests for running the pipeline on the open page from the GUI backend."""

import queue
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from PIL import Image

from collectra.gui import run_log
from collectra.gui.backend import GUIBackend
from collectra.pipelines.run_plan import RunPlan, RunReport, TaskOutcome, TaskPlan

GUARD = {"success": False, "error": "A run is in progress on this page"}
TIMEOUT = 5


class FakePipeline:
    """A pipeline whose run_page waits at every task until the test lets it go."""

    name = "demo"
    version = "1.2"
    ext = "collectra"

    def __init__(self, plan=None):
        self.plan = plan or RunPlan(
            [
                TaskPlan("ocr", "run", "no output yet"),
                TaskPlan("tidy", "rerun", "upstream ocr runs"),
                TaskPlan("tag", "skip", "already has output"),
            ]
        )
        self.plan_error = None
        self.run_error = None
        self.bad_report = None
        self.fail = set()
        self.entered = queue.Queue()
        self.gate = threading.Semaphore(0)
        self.folders = []
        self.log_at_start = None

    def plan_page_run(self, folder):
        if self.plan_error:
            raise self.plan_error
        return self.plan

    def run_page(self, folder, plan, should_stop):
        self.folders.append(folder)
        log = folder / run_log.LOG_NAME
        self.log_at_start = log.read_text() if log.exists() else None
        if self.run_error:
            raise self.run_error
        if self.bad_report is not None:
            return self.bad_report
        outcomes = []
        for task in plan.tasks:
            if task.action == "skip":
                outcomes.append(
                    TaskOutcome(task.name, "skipped", "kept existing output")
                )
                continue
            if should_stop():
                outcomes.append(
                    TaskOutcome(task.name, "cancelled", "stopped before it started")
                )
                continue
            self.entered.put(task.name)
            assert self.gate.acquire(timeout=TIMEOUT)
            if task.name in self.fail:
                outcomes.append(TaskOutcome(task.name, "failed", "RuntimeError: boom"))
                continue
            results = folder / "results.yaml"
            data = yaml.safe_load(results.read_text())
            data["words"]["data"] = f"from {task.name}"
            results.write_text(yaml.safe_dump(data))
            status = "re-ran" if task.action == "rerun" else "ran"
            detail = "(upstream ocr ran)" if status == "re-ran" else "no output yet"
            outcomes.append(TaskOutcome(task.name, status, detail))
        return RunReport(outcomes)


def make_page(root: Path, name: str) -> Path:
    folder = root / f"{name}.collectra"
    folder.mkdir()
    Image.new("RGB", (40, 30)).save(folder / "page.png")
    (folder / "results.yaml").write_text(
        yaml.safe_dump(
            {
                "page": {"type": "collectra.Image", "id": "page", "data": "page.png"},
                "words": {
                    "type": "collectra.Text",
                    "id": "words",
                    "parents": "page",
                    "data": "hello",
                },
            }
        )
    )
    return folder


@pytest.fixture
def pipeline():
    return FakePipeline()


@pytest.fixture
def page(tmp_path):
    return make_page(tmp_path, "p1")


@pytest.fixture
def backend(pipeline, page):
    backend = GUIBackend(pipeline)
    assert backend.load_yaml(str(page / "results.yaml"))["success"]
    yield backend
    # Never leave a fake run waiting on its gate.
    for _ in range(10):
        pipeline.gate.release()
    thread = backend._run_thread
    if thread is not None:
        thread.join(TIMEOUT)


def wait_started(pipeline):
    return pipeline.entered.get(timeout=TIMEOUT)


def finish(backend, pipeline, tasks=0):
    for _ in range(tasks):
        pipeline.gate.release()
    backend._run_thread.join(TIMEOUT)
    assert not backend._run_thread.is_alive()
    return backend.get_run_status()


def log_text(folder):
    return (folder / run_log.LOG_NAME).read_text(encoding="utf-8")


# Preview


def test_preview_lists_run_rerun_skip_with_reasons(backend):
    assert backend.get_run_preview() == {
        "success": True,
        "can_run": True,
        "tasks": [
            {"name": "ocr", "action": "run", "reason": "no output yet"},
            {"name": "tidy", "action": "rerun", "reason": "upstream ocr runs"},
            {"name": "tag", "action": "skip", "reason": "already has output"},
        ],
        "message": "",
    }


def test_preview_cannot_run_without_pipeline(page):
    backend = GUIBackend(None)
    assert backend.load_yaml(str(page / "results.yaml"))["success"]
    preview = backend.get_run_preview()
    assert preview["can_run"] is False
    assert preview["tasks"] == []
    assert preview["message"] == "No pipeline is loaded"
    assert backend.start_run() == {"success": False, "error": "No pipeline is loaded"}


def test_preview_cannot_run_without_page(pipeline):
    backend = GUIBackend(pipeline)
    preview = backend.get_run_preview()
    assert preview["can_run"] is False
    assert preview["message"] == "Open a page first"
    assert backend.start_run() == {"success": False, "error": "Open a page first"}


def test_preview_cannot_run_when_image_copy_missing(backend, pipeline, page):
    pipeline.plan_error = FileNotFoundError(
        f"page.png is missing from {page}; the results refer to it."
    )
    preview = backend.get_run_preview()
    assert preview["can_run"] is False
    assert preview["tasks"] == []
    assert preview["message"] == (
        f"page.png is missing from {page}; the results refer to it."
    )
    result = backend.start_run()
    assert result == {"success": False, "error": preview["message"]}
    assert pipeline.folders == []
    assert not (page / run_log.LOG_NAME).exists()
    assert backend.get_run_status()["running"] is False


def test_preview_cannot_run_when_tasks_do_not_match(backend, pipeline):
    pipeline.plan_error = ValueError("The results in p1 do not match this pipeline")
    preview = backend.get_run_preview()
    assert preview["can_run"] is False
    assert preview["message"] == "The results in p1 do not match this pipeline"


def test_preview_cannot_run_while_a_run_is_active(backend, pipeline):
    assert backend.start_run()["success"]
    wait_started(pipeline)
    preview = backend.get_run_preview()
    assert preview["can_run"] is False
    assert preview["message"] == "A run is already in progress"
    finish(backend, pipeline, tasks=2)
    assert backend.get_run_preview()["can_run"] is True


# Starting


def test_second_start_is_refused_while_running(backend, pipeline, page):
    assert backend.start_run("first") == {"success": True}
    assert wait_started(pipeline) == "ocr"
    assert backend.start_run("second") == {
        "success": False,
        "error": "A run is already in progress",
    }
    status = finish(backend, pipeline, tasks=2)
    assert status["error"] is None
    assert status["summary"] == "2 ran, 1 skipped, 0 failed"
    assert pipeline.folders == [page.resolve()]
    text = log_text(page)
    assert text.count("started") == 1
    assert "note: first" in text
    assert "second" not in text


def test_log_started_before_run_and_run_does_not_start_if_log_unwritable(
    backend, pipeline, page
):
    assert backend.start_run()["success"]
    wait_started(pipeline)
    assert "started  demo 1.2" in pipeline.log_at_start
    assert "ocr  planned to run: no output yet" in pipeline.log_at_start
    assert "result" not in pipeline.log_at_start
    finish(backend, pipeline, tasks=2)
    assert "  result  2 ran, 1 skipped, 0 failed" in log_text(page)

    pipeline.folders.clear()
    with patch.object(
        run_log, "_append", side_effect=PermissionError("log is read-only")
    ):
        result = backend.start_run()
    assert result == {"success": False, "error": "log is read-only"}
    assert pipeline.folders == []
    assert backend.get_run_status()["running"] is False
    assert backend._run_guard(page) is None
    # The lock was released: a later run can start.
    assert backend.start_run()["success"]
    wait_started(pipeline)
    finish(backend, pipeline, tasks=2)


def test_note_is_cleaned_in_log(backend, pipeline, page):
    note = "  first line\nsecond\tline  é" + "x" * 300
    assert backend.start_run(note)["success"]
    wait_started(pipeline)
    finish(backend, pipeline, tasks=2)
    text = log_text(page)
    assert f"  note: {run_log.clean_note(note)}\n" in text
    assert "first line second line é" in text
    assert "\nsecond" not in text


def test_failed_task_is_logged_as_failed(backend, pipeline, page):
    pipeline.fail = {"tidy"}
    assert backend.start_run()["success"]
    wait_started(pipeline)
    status = finish(backend, pipeline, tasks=2)
    assert status["summary"] == "1 ran, 1 skipped, 1 failed"
    assert "  tidy  failed: RuntimeError: boom" in log_text(page)


def test_run_that_crashes_is_still_recorded(backend, pipeline, page):
    pipeline.run_error = RuntimeError("the run broke")
    assert backend.start_run()["success"]
    status = finish(backend, pipeline)
    assert status["running"] is False
    assert status["finished"] is True
    assert status["error"] == "the run broke"
    assert status["summary"] == "0 ran, 1 skipped, 2 failed"
    text = log_text(page)
    assert "  ocr  failed: RuntimeError: the run broke" in text
    assert "  tag  skipped (kept existing output)" in text
    assert backend._run_guard(page) is None


def test_unwritable_log_at_the_end_keeps_results_and_reports_error(
    backend, pipeline, page
):
    assert backend.start_run()["success"]
    wait_started(pipeline)
    with patch.object(run_log, "finish_entry", side_effect=OSError("disk full")):
        status = finish(backend, pipeline, tasks=2)
    assert status["summary"] == "2 ran, 1 skipped, 0 failed"
    assert status["error"] == "The results were kept, but the run log: disk full"
    assert yaml.safe_load((page / "results.yaml").read_text())["words"]["data"] == (
        "from tidy"
    )


# Status and stop


def test_status_before_any_run(pipeline):
    assert GUIBackend(pipeline).get_run_status() == {
        "running": False,
        "finished": False,
        "folder": "",
        "current": None,
        "done": 0,
        "total": 0,
        "summary": None,
        "error": None,
    }


def test_status_reports_progress_and_summary(backend, pipeline, page):
    assert backend.start_run()["success"]
    assert wait_started(pipeline) == "ocr"
    status = backend.get_run_status()
    assert status == {
        "running": True,
        "finished": False,
        "folder": str(page.resolve()),
        "current": None,
        "done": 0,
        "total": 2,
        "summary": None,
        "error": None,
    }
    pipeline.gate.release()
    assert wait_started(pipeline) == "tidy"
    assert backend.get_run_status()["done"] == 1
    status = finish(backend, pipeline, tasks=1)
    assert status == {
        "running": False,
        "finished": True,
        "folder": str(page.resolve()),
        "current": None,
        "done": 2,
        "total": 2,
        "summary": "2 ran, 1 skipped, 0 failed",
        "error": None,
    }


def test_stop_marks_remaining_tasks_cancelled_in_log(backend, pipeline, page):
    assert backend.stop_run() == {"success": False}
    assert backend.start_run()["success"]
    wait_started(pipeline)
    assert backend.stop_run() == {"success": True}
    status = finish(backend, pipeline, tasks=1)
    assert status["summary"] == "1 ran, 1 skipped, 0 failed, 1 cancelled"
    assert status["done"] == 1
    text = log_text(page)
    assert "  ocr  ran" in text
    assert "  tidy  cancelled: stopped before it started" in text


def test_page_reloads_from_disk_when_its_run_ends(backend, pipeline):
    assert backend.start_run()["success"]
    wait_started(pipeline)
    finish(backend, pipeline, tasks=2)
    assert backend._graph.get_data("words") == "from tidy"


# The edit lock


def write_calls(results):
    return {
        "update_display_text": lambda b: b.update_display_text(
            "words", "edited", str(results)
        ),
        "update_segmentation_mask": lambda b: b.update_segmentation_mask(
            "page", [], str(results), "north"
        ),
        "rotate_image_clockwise": lambda b: b.rotate_image_clockwise("page"),
        "git_action": lambda b: b.git_action("pull"),
        "update_page_text": lambda b: b.update_page_text("markdown", "# edited"),
        "update_node_data": lambda b: b.update_node_data("words", "edited"),
        "update_node_coordinates": lambda b: b.update_node_coordinates(
            "words",
            {
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.1,
                "height_relative": 0.1,
            },
        ),
        "create_annotation": lambda b: b.create_annotation(
            {
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.1,
                "height_relative": 0.1,
            },
            "box",
            "page",
            str(results.parent / "page.png"),
        ),
        "delete_nested": lambda b: b.delete_nested("page"),
        "delete_annotation": lambda b: b.delete_annotation("words"),
        "rename_annotation": lambda b: b.rename_annotation("words", "renamed"),
        "set_link_name": lambda b: b.set_link_name("words", "other"),
        "create_link": lambda b: b.create_link("page", "handwritten"),
    }


def test_writes_blocked_during_run(backend, pipeline, page):
    results = page / "results.yaml"
    before = results.read_bytes()
    assert backend.start_run()["success"]
    wait_started(pipeline)
    for name, call in write_calls(results).items():
        assert call(backend) == GUARD, name
    assert backend.set_table_cells([])["success"]
    assert results.read_bytes() == before
    assert not (page / "markdown.md").exists()
    finish(backend, pipeline, tasks=2)

    with patch("collectra.gui.source_control.run_git", return_value={"ok": 1}):
        for name, call in write_calls(results).items():
            assert call(backend) != GUARD, name
    assert backend.update_page_text("markdown", "# again") == {"success": True}
    assert (page / "markdown.md").read_text() == "# again"


def test_table_edit_of_the_running_page_is_skipped(tmp_path, pipeline, page):
    backend = GUIBackend(pipeline)
    backend._collectra_folders = [
        {"name": page.name, "path": str(page), "yaml_path": str(page / "results.yaml")}
    ]
    assert backend.load_yaml(str(page / "results.yaml"))["success"]
    assert backend.start_run()["success"]
    wait_started(pipeline)
    with patch.object(backend, "_page_text_nodes", return_value={"words": ["words"]}):
        result = backend.set_table_cells(
            [{"index": 0, "label": "words", "text": "typed"}]
        )
    assert result["saved"] == []
    assert result["skipped"] == [
        {"index": 0, "label": "words", "reason": GUARD["error"]}
    ]
    finish(backend, pipeline, tasks=2)


def test_navigating_to_another_page_keeps_run_on_original_folder(
    tmp_path, backend, pipeline, page
):
    other = make_page(tmp_path, "p2")
    assert backend.start_run("on p1")["success"]
    wait_started(pipeline)
    assert backend.load_yaml(str(other / "results.yaml"))["success"]

    # The second page stays editable; the running one does not.
    assert backend.update_display_text("words", "p2 edit") == {
        "success": True,
        "changed": True,
    }
    assert backend.update_page_text("markdown", "# p2") == {"success": True}
    assert backend.git_action("status") != GUARD
    assert (
        backend.update_display_text("words", "p1 edit", str(page / "results.yaml"))
        == GUARD
    )
    assert backend._run_guard(page) == GUARD
    assert backend._run_guard(other) is None
    assert backend.get_run_status()["folder"] == str(page.resolve())

    status = finish(backend, pipeline, tasks=2)
    assert pipeline.folders == [page.resolve()]
    assert status["folder"] == str(page.resolve())
    assert "note: on p1" in log_text(page)
    assert not (other / run_log.LOG_NAME).exists()
    assert yaml.safe_load((page / "results.yaml").read_text())["words"]["data"] == (
        "from tidy"
    )
    p2 = yaml.safe_load((other / "results.yaml").read_text())
    assert p2["words"]["data"] == "p2 edit"
    # The open page is still the second one, and it was not reloaded from p1.
    assert backend._yaml_path == str(other / "results.yaml")
    assert backend._graph.get_data("words") == "p2 edit"


def test_run_guard_without_folder_uses_open_page(backend, pipeline, page):
    assert backend._run_guard() is None
    assert backend.start_run()["success"]
    wait_started(pipeline)
    assert backend._run_guard() == GUARD
    finish(backend, pipeline, tasks=2)
    assert backend._run_guard() is None
    assert GUIBackend(pipeline)._run_guard() is None


# One shared lock, atomic with every write


def page_writers(results):
    return {k: v for k, v in write_calls(results).items() if k != "git_action"}


@pytest.mark.parametrize("name", sorted(page_writers(Path("r/results.yaml"))))
def test_every_page_writer_holds_the_shared_edit_lock(backend, page, name):
    call = page_writers(page / "results.yaml")[name]
    backend._text_save_lock.acquire()
    try:
        thread = threading.Thread(target=call, args=(backend,), daemon=True)
        thread.start()
        thread.join(0.1)
        assert thread.is_alive(), f"{name} wrote without the shared lock"
    finally:
        backend._text_save_lock.release()
    thread.join(TIMEOUT)
    assert not thread.is_alive()


def test_start_run_waits_for_a_write_in_progress(backend, pipeline, page):
    inside, release = threading.Event(), threading.Event()
    save = backend._save_collectra_file

    def slow_save():
        inside.set()
        assert release.wait(TIMEOUT)
        save()

    results = {}
    with patch.object(backend, "_save_collectra_file", side_effect=slow_save):
        writer = threading.Thread(
            target=lambda: results.update(
                edit=backend.update_node_data("words", "typed")
            ),
            daemon=True,
        )
        writer.start()
        assert inside.wait(TIMEOUT)
        starter = threading.Thread(
            target=lambda: results.update(start=backend.start_run()), daemon=True
        )
        starter.start()
        starter.join(0.1)
        assert starter.is_alive(), "start_run did not wait for the edit"
        assert pipeline.folders == []
        assert not (page / run_log.LOG_NAME).exists()
        release.set()
        writer.join(TIMEOUT)
        starter.join(TIMEOUT)
    assert results["edit"]["success"]
    assert results["start"] == {"success": True}
    wait_started(pipeline)
    finish(backend, pipeline, tasks=2)
    assert pipeline.folders == [page.resolve()]


# Git


def test_start_run_is_refused_while_git_runs(backend, pipeline, page):
    backend._git_lock.acquire()
    try:
        result = backend.start_run()
    finally:
        backend._git_lock.release()
    assert result == {
        "success": False,
        "error": "A Git operation is running; try again when it finishes",
    }
    assert pipeline.folders == []
    assert not (page / run_log.LOG_NAME).exists()
    assert backend.get_run_status()["running"] is False
    assert backend.start_run()["success"]
    wait_started(pipeline)
    finish(backend, pipeline, tasks=2)


@pytest.mark.parametrize("action", ["pull", "save", "merge", "abort", "push"])
def test_changing_git_actions_are_refused_on_every_page_during_a_run(
    tmp_path, backend, pipeline, action
):
    other = make_page(tmp_path, "p2")
    assert backend.start_run()["success"]
    wait_started(pipeline)
    assert backend.load_yaml(str(other / "results.yaml"))["success"]
    with patch("collectra.gui.source_control.run_git") as run_git:
        assert backend.git_action(action, "message") == GUARD
    run_git.assert_not_called()
    finish(backend, pipeline, tasks=2)
    with patch("collectra.gui.source_control.run_git", return_value={"ok": 1}):
        assert backend.git_action(action, "message") == {"ok": 1}


def test_git_status_stays_allowed_during_a_run(backend, pipeline):
    assert backend.start_run()["success"]
    wait_started(pipeline)
    with patch("collectra.gui.source_control.run_git", return_value={"ok": 1}):
        assert backend.git_action("status") == {"ok": 1}
    finish(backend, pipeline, tasks=2)


# Ending a run safely


def test_failed_reload_refuses_edits_until_the_page_is_reloaded(
    backend, pipeline, page
):
    assert backend.start_run()["success"]
    wait_started(pipeline)
    with patch.object(
        backend, "_load_yaml", return_value={"success": False, "error": "bad yaml"}
    ):
        status = finish(backend, pipeline, tasks=2)
    assert status["error"] == ("The page could not be reloaded after the run: bad yaml")
    stale = {
        "success": False,
        "error": "The run changed this page; open it again before editing",
    }
    for name, call in page_writers(page / "results.yaml").items():
        assert call(backend) == stale, name
    assert yaml.safe_load((page / "results.yaml").read_text())["words"]["data"] == (
        "from tidy"
    )
    assert backend.load_yaml(str(page / "results.yaml"))["success"]
    assert backend._run_guard() is None
    assert backend.update_node_data("words", "after")["success"]


def test_a_run_page_result_that_is_not_a_report_still_ends_the_run(
    backend, pipeline, page
):
    pipeline.bad_report = "not a report"
    assert backend.start_run()["success"]
    status = finish(backend, pipeline)
    assert status["running"] is False
    assert status["error"] == "run_page returned str, not a RunReport"
    assert status["summary"] == "0 ran, 1 skipped, 2 failed"
    assert "  ocr  failed: TypeError: run_page returned str" in log_text(page)
    assert backend._run_guard(page) is None
    assert not backend._run_lock.locked()


def test_an_error_while_ending_the_run_still_releases_the_lock(backend, pipeline, page):
    assert backend.start_run()["success"]
    wait_started(pipeline)
    with patch.object(run_log, "summarize", side_effect=RuntimeError("oops")):
        status = finish(backend, pipeline, tasks=2)
    assert status["running"] is False
    assert status["error"] == "The run could not be recorded: oops"
    assert not backend._run_lock.locked()
    assert backend.start_run()["success"]
    wait_started(pipeline)
    finish(backend, pipeline, tasks=2)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_base_exception_in_the_run_is_recorded_and_re_raised(backend, pipeline, page):
    pipeline.run_error = SystemExit("shutting down")
    assert backend.start_run()["success"]
    status = finish(backend, pipeline)
    assert status["running"] is False
    assert status["error"] == "shutting down"
    assert status["summary"] == "0 ran, 1 skipped, 2 failed"
    assert "  ocr  failed: SystemExit: shutting down" in log_text(page)
    assert not backend._run_lock.locked()
