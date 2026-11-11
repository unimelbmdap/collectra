"""Tests for running every page of the collection from the GUI backend."""

import shutil
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from collectra.gui import run_log
from collectra.gui.backend import GUIBackend, _RunState
from collectra.pipelines.run_plan import RunPlan, RunReport, TaskOutcome, TaskPlan
from tests.gui.test_backend_run import GUARD, TIMEOUT, log_text, make_page

NAMES = ("ocr", "tidy", "tag")
PLAN = RunPlan(
    [
        TaskPlan("ocr", "run", "no output yet"),
        TaskPlan("tidy", "rerun", "upstream ocr runs"),
        TaskPlan("tag", "skip", "already has output"),
    ]
)
COMPLETE = RunPlan([TaskPlan(name, "skip", "already has output") for name in NAMES])
NEW = RunPlan([TaskPlan(name, "run", "no output yet") for name in NAMES])


class FakePipeline:
    """A pipeline whose run_pages waits until the test lets it go."""

    name = "demo"
    version = "1.2"
    ext = "collectra"

    def __init__(self):
        self.plans = {"p2.collectra": COMPLETE, "p3.collectra": NEW}
        self.plan_errors = {}
        self.run_error = None
        self.fail = set()
        self.calls = []
        self.logs_at_start = {}
        self.on_page = lambda: None
        self.started = threading.Event()
        self.gate = threading.Event()

    def plan_page_run(self, folder, overwrite=False):
        if folder.name in self.plan_errors:
            raise self.plan_errors[folder.name]
        plan = self.plans.get(folder.name, PLAN)
        if overwrite:
            return RunPlan(
                [
                    task if task.action == "run" else TaskPlan(task.name, "rerun", "x")
                    for task in plan.tasks
                ]
            )
        return plan

    def run_pages(self, plans, rerun, should_stop, page_done):
        self.calls.append((dict(plans), rerun))
        self.logs_at_start = {folder: log_text(folder) for folder in plans}
        self.started.set()
        assert self.gate.wait(TIMEOUT)
        if self.run_error:
            raise self.run_error
        reports = {}
        for folder, plan in plans.items():
            outcomes = []
            for task in plan.tasks:
                if task.action == "skip":
                    outcome = ("skipped", "kept existing output")
                elif should_stop():
                    outcome = ("cancelled", "stopped before it started")
                elif (folder.name, task.name) in self.fail:
                    outcome = ("failed", "RuntimeError: boom")
                elif task.action == "rerun":
                    outcome = ("re-ran", "replaced existing output")
                else:
                    outcome = ("ran", task.reason)
                outcomes.append(TaskOutcome(task.name, *outcome))
            reports[folder] = RunReport(outcomes)
            page_done(folder)
            self.on_page()
        return reports


@pytest.fixture
def pipeline():
    return FakePipeline()


@pytest.fixture
def pages(tmp_path):
    return [make_page(tmp_path, name).resolve() for name in ("p1", "p2", "p3")]


def collection(pipeline, pages):
    backend = GUIBackend(pipeline)
    backend._collectra_folders = [
        {"name": page.name, "path": str(page)} for page in pages
    ]
    return backend


@pytest.fixture
def backend(pipeline, pages):
    backend = collection(pipeline, pages)
    assert backend.load_yaml(str(pages[0] / "results.yaml"))["success"]
    yield backend
    pipeline.gate.set()
    if backend._run_thread is not None:
        backend._run_thread.join(TIMEOUT)


def finish(backend, pipeline):
    pipeline.gate.set()
    backend._run_thread.join(TIMEOUT)
    assert not backend._run_thread.is_alive()
    return backend.get_run_status()


def start(backend, pipeline, **kwargs):
    assert backend.start_collection_run(**kwargs) == {"success": True}
    assert pipeline.started.wait(TIMEOUT)


# Preview


def test_preview_counts_pages_and_tasks(backend):
    assert backend.get_collection_preview() == {
        "success": True,
        "can_run": True,
        "pages": 3,
        "to_replace": 2,
        "first_time": 1,
        "to_run": 2,
        "tasks": [
            {"name": "ocr", "pages": 2, "replace": 0},
            {"name": "tidy", "pages": 1, "replace": 0},
            {"name": "tag", "pages": 1, "replace": 0},
        ],
        "unplannable": [],
        "message": "",
    }


def test_preview_with_overwrite_counts_what_is_replaced(backend):
    preview = backend.get_collection_preview(overwrite=True)
    assert (preview["pages"], preview["to_replace"], preview["first_time"]) == (
        3,
        2,
        1,
    )
    assert preview["tasks"] == [
        {"name": "ocr", "pages": 3, "replace": 1},
        {"name": "tidy", "pages": 3, "replace": 2},
        {"name": "tag", "pages": 3, "replace": 2},
    ]


def test_tasks_are_listed_in_pipeline_order(backend, pipeline):
    pipeline.plans = {
        "p1.collectra": RunPlan([*COMPLETE.tasks[:2], TaskPlan("tag", "run", "")]),
        "p2.collectra": NEW,
        "p3.collectra": COMPLETE,
    }
    names = [task["name"] for task in backend.get_collection_preview()["tasks"]]
    assert names == list(NAMES)


def test_unplannable_pages_are_listed_and_left_out(backend, pipeline, pages):
    pipeline.plan_errors = {"p2.collectra": FileNotFoundError("page.png is missing")}
    preview = backend.get_collection_preview()
    assert preview["unplannable"] == [
        {"name": "p2.collectra", "reason": "page.png is missing"}
    ]
    assert (preview["pages"], preview["to_replace"], preview["first_time"]) == (
        3,
        1,
        1,
    )
    start(backend, pipeline)
    finish(backend, pipeline)
    assert list(pipeline.calls[0][0]) == [pages[0], pages[2]]
    assert not (pages[1] / run_log.LOG_NAME).exists()
    assert backend.get_run_status()["total"] == 2


def all_complete(backend):
    backend._pipeline.plans = dict.fromkeys(
        ("p1.collectra", "p2.collectra", "p3.collectra"), COMPLETE
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda b: setattr(b, "_pipeline", None), "No pipeline is loaded"),
        (lambda b: setattr(b, "_collectra_folders", []), "Open a page first"),
        (lambda b: b._run_lock.acquire(), "A run is already in progress"),
        (all_complete, "Every page that can run already has its results"),
    ],
)
def test_preview_and_start_refuse_when_nothing_can_run(backend, change, message):
    change(backend)
    preview = backend.get_collection_preview()
    assert preview["can_run"] is False
    assert preview["message"] == message
    assert backend.start_collection_run() == {"success": False, "error": message}
    assert backend.get_run_status()["running"] is False


def test_preview_holds_the_run_lock_while_it_plans(backend, pipeline):
    held = []
    plan = pipeline.plan_page_run

    def plan_page_run(folder, overwrite=False):
        held.append(backend._run_lock.locked())
        return plan(folder, overwrite)

    pipeline.plan_page_run = plan_page_run
    assert backend.get_collection_preview()["can_run"] is True
    assert held == [True, True, True]
    assert not backend._run_lock.locked()


# Starting and logging


def test_unticked_run_skips_what_has_output_and_logs_each_page(
    backend, pipeline, pages
):
    start(backend, pipeline, note="  first\n pass ")
    plans, rerun = pipeline.calls[0]
    assert rerun == set()
    assert plans[pages[0]].tasks[1] == TaskPlan("tidy", "skip", "already has output")
    assert "  tidy  planned to skip: already has output\n" in (
        pipeline.logs_at_start[pages[0]]
    )
    status = finish(backend, pipeline)
    assert status["summary"] == "3 pages: 4 ran, 5 skipped, 0 failed"
    assert status["error"] is None
    first = log_text(pages[0])
    assert "  note: first pass\n" in first
    assert "  ocr  ran\n  tidy  skipped (kept existing output)\n" in first
    assert "  result  1 ran, 2 skipped, 0 failed, " in first
    assert "  result  0 ran, 3 skipped, 0 failed, " in log_text(pages[1])
    assert "  result  3 ran, 0 skipped, 0 failed, " in log_text(pages[2])


def test_ticked_run_reruns_every_task_and_logs_the_replacing(backend, pipeline, pages):
    start(backend, pipeline, overwrite=True)
    assert pipeline.calls[0][1] == set(NAMES)
    finish(backend, pipeline)
    assert "  tag  re-ran, replaced existing output\n" in log_text(pages[1])


def test_one_failing_page_does_not_stop_the_others(backend, pipeline, pages):
    pipeline.fail = {("p1.collectra", "ocr")}
    start(backend, pipeline)
    status = finish(backend, pipeline)
    assert status["summary"] == "3 pages: 3 ran, 5 skipped, 1 failed"
    assert "  ocr  failed: RuntimeError: boom\n" in log_text(pages[0])
    assert "  result  3 ran, 0 skipped, 0 failed, " in log_text(pages[2])


def test_a_log_that_cannot_be_written_starts_nothing(backend, pipeline, pages):
    (pages[1] / run_log.LOG_NAME).mkdir()
    result = backend.start_collection_run()
    assert result["success"] is False
    assert result["error"].startswith(
        "The run log of p2.collectra could not be written: "
    )
    assert pipeline.calls == []
    assert backend.get_run_status()["running"] is False
    assert not backend._run_lock.locked()
    assert backend._run_guard(pages[0]) is None
    # The page already started logs that the run did not start
    assert log_text(pages[0]).endswith(
        "  ocr  cancelled: the run did not start\n"
        "  tidy  cancelled: the run did not start\n"
        "  tag  cancelled: the run did not start\n"
        "  result  0 ran, 0 skipped, 0 failed, 3 cancelled, 0.0 s\n"
    )


def test_a_finish_entry_that_cannot_be_written_keeps_the_results(
    backend, pipeline, pages
):
    start(backend, pipeline)
    (pages[2] / run_log.LOG_NAME).unlink()
    (pages[2] / run_log.LOG_NAME).mkdir()
    status = finish(backend, pipeline)
    assert status["error"].startswith(
        "The results were kept, but the run log of p3.collectra: "
    )
    assert "  result  1 ran, 2 skipped" in log_text(pages[0])
    assert not backend._run_lock.locked()


def test_start_is_refused_while_git_runs(backend, pipeline, pages):
    backend._git_lock.acquire()
    try:
        result = backend.start_collection_run()
    finally:
        backend._git_lock.release()
    assert result == {
        "success": False,
        "error": "A Git operation is running; try again when it finishes",
    }
    assert pipeline.calls == []
    assert not (pages[0] / run_log.LOG_NAME).exists()


def test_start_waits_for_an_edit_in_progress(backend, pipeline):
    results = []
    backend._text_save_lock.acquire()
    try:
        starter = threading.Thread(
            target=lambda: results.append(backend.start_collection_run()),
            daemon=True,
        )
        starter.start()
        starter.join(0.1)
        assert starter.is_alive(), "the run did not wait for the edit"
        assert backend._run_state is None
    finally:
        backend._text_save_lock.release()
    starter.join(TIMEOUT)
    assert results == [{"success": True}]
    finish(backend, pipeline)


# While it runs


def test_progress_counts_pages(backend, pipeline, pages):
    seen = []
    pipeline.on_page = lambda: seen.append(backend.get_run_status()["done"])
    start(backend, pipeline)
    status = backend.get_run_status()
    assert (status["unit"], status["done"], status["total"]) == ("pages", 0, 3)
    assert status["folders"] == sorted(map(str, pages))
    assert status["running"] is True
    status = finish(backend, pipeline)
    assert seen == [1, 2, 3]
    assert (status["done"], status["total"], status["running"]) == (3, 3, False)


def test_every_page_of_the_run_is_locked_and_only_those(
    tmp_path, backend, pipeline, pages
):
    pipeline.plan_errors = {"p3.collectra": ValueError("not this pipeline")}
    outside = make_page(tmp_path, "p4")
    start(backend, pipeline)
    assert backend._run_guard(pages[0]) == GUARD
    assert backend._run_guard(pages[1]) == GUARD
    assert backend._run_guard(pages[2]) is None
    assert backend.update_node_data("words", "typed") == GUARD
    assert backend.load_yaml(str(outside / "results.yaml"))["success"]
    assert backend._run_guard() is None
    assert backend.update_node_data("words", "typed")["success"]
    finish(backend, pipeline)
    assert backend._run_guard(pages[0]) is None


@pytest.mark.parametrize("action", ["pull", "save", "merge", "abort", "push"])
def test_changing_git_actions_are_refused(backend, pipeline, action):
    start(backend, pipeline)
    with patch("collectra.gui.source_control.run_git") as run_git:
        assert backend.git_action(action, "message") == GUARD
    run_git.assert_not_called()
    finish(backend, pipeline)


def test_restart_and_update_are_refused(backend, pipeline):
    backend._window = Mock()
    start(backend, pipeline)
    assert backend.restart_gui()["success"] is False
    backend._window.destroy.assert_not_called()
    with patch("collectra.gui.updates.update_repository") as update:
        assert backend.install_update("pipeline")["success"] is False
    update.assert_not_called()
    finish(backend, pipeline)


def test_only_one_run_at_a_time(backend, pipeline):
    start(backend, pipeline)
    busy = {"success": False, "error": "A run is already in progress"}
    assert backend.start_collection_run() == busy
    assert backend.start_run() == busy
    assert backend.get_run_preview()["can_run"] is False
    assert backend.get_collection_preview()["can_run"] is False
    finish(backend, pipeline)
    assert len(pipeline.calls) == 1


def test_stop_ends_the_run_and_logs_cancelled(backend, pipeline, pages):
    start(backend, pipeline)
    assert backend.stop_run() == {"success": True}
    status = finish(backend, pipeline)
    assert status["summary"] == "3 pages: 0 ran, 5 skipped, 0 failed, 4 cancelled"
    assert "  ocr  cancelled: stopped before it started\n" in log_text(pages[0])
    assert "  ocr  cancelled: stopped before it started\n" in log_text(pages[2])


def test_a_crash_records_failures_and_releases_the_lock(backend, pipeline, pages):
    pipeline.run_error = RuntimeError("the run broke")
    start(backend, pipeline)
    status = finish(backend, pipeline)
    assert status["running"] is False
    assert status["error"] == "the run broke"
    assert status["summary"] == "3 pages: 0 ran, 5 skipped, 4 failed"
    assert "  ocr  failed: RuntimeError: the run broke\n" in log_text(pages[0])
    assert "  tag  skipped (kept existing output)\n" in log_text(pages[0])
    assert "  tag  failed: RuntimeError: the run broke\n" in log_text(pages[2])
    assert not backend._run_lock.locked()
    assert backend._run_guard(pages[0]) is None


def test_the_open_page_is_reloaded_when_the_run_ends(backend, pipeline, pages):
    start(backend, pipeline)
    results = pages[0] / "results.yaml"
    data = yaml.safe_load(results.read_text())
    data["words"]["data"] = "from the run"
    results.write_text(yaml.safe_dump(data))
    finish(backend, pipeline)
    assert backend._graph.get_data("words") == "from the run"


def test_the_state_resolves_its_folders(tmp_path):
    page = make_page(tmp_path, "a")
    state = _RunState(frozenset({tmp_path / "x" / ".." / page.name}), total=1)
    assert state.folders == frozenset({page.resolve()})


# A real pipeline, end to end


@pytest.fixture
def real_pages(tmp_path, raw_img_path):
    """Four pages every task has run on once."""
    from tests.unit.test_run_page import RAN, build

    source = tmp_path / "input"
    source.mkdir()
    for name in "abcd":
        shutil.copy(raw_img_path, source / f"{name}.jpg")
    build(tmp_path).run(files=[str(source / f"{name}.jpg") for name in "abcd"])
    RAN.clear()
    return [source / f"{name}.pge" for name in "abcd"]


def real_collection(tmp_path, pages):
    from tests.unit.test_run_page import build

    return collection(build(tmp_path), pages)


def outputs(folder: Path) -> dict:
    """Each task output's results entry and the bytes of its file."""
    data = yaml.safe_load((folder / "results.yaml").read_text())
    return {
        name: (data[name], (folder / data[name]["path"]).read_bytes())
        for name in ("boxes", "words", "label")
        if name in data
    }


def run_to_the_end(backend, **kwargs):
    assert backend.start_collection_run(**kwargs) == {"success": True}
    backend._run_thread.join(TIMEOUT)
    return backend.get_run_status()


def test_unticked_collection_run_replaces_nothing(tmp_path, real_pages):
    from tests.unit.test_run_page import RAN, drop

    a, b, c, d = real_pages
    drop(a, "words")
    drop(c, "boxes", "words", "label")
    (d / "d.jpg").unlink()
    before = {page: outputs(page) for page in real_pages}
    backend = real_collection(tmp_path, real_pages)
    preview = backend.get_collection_preview()
    assert preview["unplannable"] == [
        {"name": "d.pge", "reason": f"d.jpg is missing from {d.resolve()}; "
         "the results refer to it."}
    ]  # fmt: skip
    assert (preview["pages"], preview["to_replace"], preview["first_time"]) == (
        4,
        2,
        1,
    )
    assert preview["tasks"] == [
        {"name": "detect", "pages": 1, "replace": 0},
        {"name": "ocr", "pages": 2, "replace": 0},
        {"name": "classify", "pages": 1, "replace": 0},
    ]
    status = run_to_the_end(backend)
    assert status["summary"] == "3 pages: 4 ran, 5 skipped, 0 failed"
    assert sorted(RAN) == ["classify", "detect", "ocr", "ocr"]
    for page in (a, b):
        after = outputs(page)
        assert {name: after[name] for name in before[page]} == before[page]
    assert outputs(d) == before[d]
    log = log_text(a)
    assert "  classify  planned to skip: already has output\n" in log
    assert "  ocr  ran\n  classify  skipped (kept existing output)\n" in log
    assert "  result  0 ran, 3 skipped, 0 failed, " in log_text(b)
    assert "  result  3 ran, 0 skipped, 0 failed, " in log_text(c)
    assert not (d / run_log.LOG_NAME).exists()


def test_preview_counts_the_pages_with_tasks_to_run(tmp_path, real_pages):
    from tests.unit.test_run_page import drop

    a, b, c, d = real_pages
    drop(a, "words")
    drop(c, "boxes", "words", "label")
    drop(d, "label")
    backend = real_collection(tmp_path, real_pages)
    # The complete page b has nothing to run unless the tick is on
    assert backend.get_collection_preview()["to_run"] == 3
    assert backend.get_collection_preview(overwrite=True)["to_run"] == 4
    all_done = real_collection(tmp_path, [b])
    assert all_done.get_collection_preview()["to_run"] == 0


def test_ticked_collection_run_replaces_and_keeps_what_fails(
    tmp_path, real_pages, monkeypatch
):
    from tests.unit.test_run_page import Detect, fail_saving, saved

    def broken(self, image):
        raise RuntimeError("out of memory")

    pages = real_pages[:2]
    old = {page: {name: saved(page, name) for name in ("boxes", "words", "label")}
           for page in pages}  # fmt: skip
    backend = real_collection(tmp_path, pages)
    assert backend.get_collection_preview(overwrite=True)["tasks"] == [
        {"name": name, "pages": 2, "replace": 2}
        for name in ("detect", "ocr", "classify")
    ]
    monkeypatch.setattr(Detect, "run", broken)
    fail_saving(monkeypatch, "label")
    status = run_to_the_end(backend, overwrite=True)
    assert status["summary"] == "2 pages: 2 ran, 0 skipped, 4 failed"
    for page in pages:
        assert saved(page, "boxes") == old[page]["boxes"]
        assert saved(page, "label") == old[page]["label"]
        assert saved(page, "words") != old[page]["words"]
        log = log_text(page)
        assert "  ocr  re-ran, replaced existing output\n" in log
        assert "  detect  failed: RuntimeError: out of memory; kept previous " in log
        assert (
            "  classify  failed: could not save its output: OSError: disk full" in log
        )


def test_a_failing_page_and_an_unplannable_page_do_not_stop_the_others(
    tmp_path, real_pages, monkeypatch
):
    from tests.unit.test_run_page import Ocr, drop, saved
    from tests.unit.test_run_pages import on_page

    a, b, c, d = real_pages
    for page in (a, b):
        drop(page, "words", "label")
    (c / "results.yaml").write_text("not: [valid")
    monkeypatch.setattr(Ocr, "run", on_page("a.pge", Ocr))
    status = run_to_the_end(real_collection(tmp_path, real_pages))
    assert status["total"] == 3
    assert "  ocr  failed: RuntimeError: cannot read a.pge" in log_text(a)
    assert "  result  2 ran, 1 skipped, 0 failed, " in log_text(b)
    assert saved(b, "label").startswith("label of words")
    assert not (c / run_log.LOG_NAME).exists()
    assert "  result  0 ran, 3 skipped, 0 failed, " in log_text(d)


def test_an_unplannable_reason_is_the_first_line_of_the_error(backend, pipeline):
    pipeline.plan_errors = {"p2.collectra": ValueError("bad results\n  in line 3")}
    assert backend.get_collection_preview()["unplannable"] == [
        {"name": "p2.collectra", "reason": "bad results"}
    ]


def test_when_no_page_can_be_planned_the_message_says_so(backend, pipeline):
    pipeline.plan_errors = dict.fromkeys(
        ("p1.collectra", "p2.collectra", "p3.collectra"), ValueError("no")
    )
    message = "None of the pages can be run"
    preview = backend.get_collection_preview()
    assert (preview["can_run"], preview["message"]) == (False, message)
    assert backend.start_collection_run() == {"success": False, "error": message}
