"""Tests for overwriting results when running one image, and the run state."""

from unittest.mock import patch

import pytest
import yaml

from collectra.gui import run_log
from collectra.gui.backend import GUIBackend, _RunState
from tests.gui.test_backend_run import (  # noqa: F401 (fixtures)
    GUARD,
    TIMEOUT,
    FakePipeline,
    backend,
    finish,
    make_page,
    page,
    real_backend,
    real_page,
    wait_started,
)


class RecordingPipeline(FakePipeline):
    """A fake pipeline that records the overwrite each plan was asked for."""

    def __init__(self):
        super().__init__()
        self.asked = []

    def plan_page_run(self, folder, overwrite=False):
        self.asked.append(overwrite)
        return super().plan_page_run(folder)


@pytest.fixture
def pipeline():
    return RecordingPipeline()


# Overwrite on a real page


def test_overwrite_previews_every_task_with_output_as_rerun(tmp_path, real_page):
    backend = real_backend(tmp_path, real_page)
    assert [task["action"] for task in backend.get_run_preview()["tasks"]] == [
        "skip",
        "skip",
    ]
    assert backend.get_run_preview(overwrite=True)["tasks"] == [
        {"name": "describe", "action": "rerun", "reason": "overwrite"},
        {"name": "shout", "action": "rerun", "reason": "overwrite"},
    ]


def test_overwrite_runs_every_task_and_logs_the_replaced_output(tmp_path, real_page):
    from tests.unit.test_run_plan import RAN

    backend = real_backend(tmp_path, real_page)
    assert backend.start_run(overwrite=True)["success"]
    backend._run_thread.join(TIMEOUT)
    status = backend.get_run_status()
    assert status["summary"] == "2 ran, 0 skipped, 0 failed"
    assert status["error"] is None
    assert RAN == ["describe", "shout"]
    log = (real_page / run_log.LOG_NAME).read_text()
    assert "  describe  re-ran, replaced existing output\n" in log
    assert "  shout  re-ran, replaced existing output\n" in log


def test_a_run_without_overwrite_replaces_nothing(tmp_path, real_page):
    from tests.unit.test_run_plan import RAN

    def outputs():
        data = yaml.safe_load((real_page / "results.yaml").read_text())
        return data["described"], data["shouted"]

    backend = real_backend(tmp_path, real_page)
    before = outputs()
    assert backend.start_run()["success"]
    backend._run_thread.join(TIMEOUT)
    assert backend.get_run_status()["summary"] == "0 ran, 2 skipped, 0 failed"
    assert RAN == []
    assert outputs() == before


# The overwrite choice reaches the plan


def test_overwrite_is_off_unless_asked(backend, pipeline):
    backend.get_run_preview()
    assert backend.start_run(note="n")["success"]
    finish(backend, pipeline, tasks=2)
    backend.get_run_preview(overwrite=True)
    assert backend.start_run(overwrite=True)["success"]
    finish(backend, pipeline, tasks=2)
    assert pipeline.asked == [False, False, True, True]


# The run state for a set of folders


def test_a_one_folder_state_keeps_its_folder(page):
    state = _RunState(page, total=1)
    assert state.folders == frozenset({page})
    assert state.folder == page
    assert state.unit == "tasks"


def test_a_state_of_several_folders_has_no_single_folder(tmp_path):
    first, second = make_page(tmp_path, "a"), make_page(tmp_path, "b")
    state = _RunState(frozenset({first, second}), total=2, unit="pages")
    assert state.folders == frozenset({first, second})
    assert state.folder is None


def test_the_guard_refuses_every_folder_of_the_run_and_only_those(tmp_path, pipeline):
    first, second, other = (make_page(tmp_path, name) for name in "abc")
    backend = GUIBackend(pipeline)
    backend._run_state = _RunState(
        frozenset({first.resolve(), second.resolve()}), total=2
    )
    assert backend._run_guard(first) == GUARD
    assert backend._run_guard(second) == GUARD
    assert backend._run_guard(other) is None
    backend._run_state.running = False
    assert backend._run_guard(first) is None


def test_status_of_one_image_is_as_before(pipeline, page):
    backend = GUIBackend(pipeline)
    backend._run_state = _RunState(page, total=1)
    assert backend.get_run_status() == {
        "running": True,
        "finished": False,
        "folder": str(page),
        "current": None,
        "done": 0,
        "total": 1,
        "summary": None,
        "error": None,
    }


def test_status_of_pages_carries_the_folders_and_unit(tmp_path, pipeline):
    first, second = (make_page(tmp_path, name).resolve() for name in "ab")
    backend = GUIBackend(pipeline)
    backend._run_state = _RunState(frozenset({second, first}), total=2, unit="pages")
    assert backend.get_run_status() == {
        "running": True,
        "finished": False,
        "folder": "",
        "current": None,
        "done": 0,
        "total": 2,
        "summary": None,
        "error": None,
        "folders": [str(first), str(second)],
        "unit": "pages",
    }


@pytest.mark.parametrize("included, reloaded", [(True, True), (False, False)])
def test_the_open_page_reloads_when_it_is_one_of_the_folders(
    tmp_path, backend, page, included, reloaded
):
    other = make_page(tmp_path, "other").resolve()
    folders = {other, page.resolve()} if included else {other}
    state = _RunState(frozenset(folders), total=2)
    results = page / "results.yaml"
    data = yaml.safe_load(results.read_text())
    data["words"]["data"] = "changed on disk"
    results.write_text(yaml.safe_dump(data))
    with patch.object(backend, "_load_yaml", wraps=backend._load_yaml) as load:
        backend._reload_after_run(state)
    assert load.called is reloaded
    assert (backend._graph.get_data("words") == "changed on disk") is reloaded
