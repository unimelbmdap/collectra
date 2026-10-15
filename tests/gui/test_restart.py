import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from collectra.gui import backend


def test_restart_requests_close_and_preserves_open_folders(tmp_path, monkeypatch):
    api = backend.GUIBackend(SimpleNamespace(path=tmp_path, ext="collectra"))
    api._window = Mock()
    api._collectra_folders = [{"path": str(tmp_path / "page.collectra")}]
    save = Mock()
    monkeypatch.setattr(api, "_save_folder_session", save)
    assert api.restart_gui()["success"]
    save.assert_called_once()
    assert api._restart_requested
    api._window.destroy.assert_called_once()


def test_restart_failure_keeps_process_running(tmp_path):
    api = backend.GUIBackend(SimpleNamespace(path=tmp_path, ext="collectra"))
    api._window = Mock()
    api._window.destroy.side_effect = RuntimeError("Cannot close")
    assert not api.restart_gui()["success"]
    assert not api._restart_requested


@pytest.mark.parametrize("restart", [False, True])
def test_process_replaced_only_after_requested_gui_shutdown(tmp_path, monkeypatch, restart):
    import os

    api = Mock(_restart_requested=restart)
    monkeypatch.setattr(backend, "GUIBackend", Mock(return_value=api))
    monkeypatch.setattr(backend, "_set_macos_app_name", Mock())
    monkeypatch.setattr(backend.webview, "create_window", Mock())
    events = []
    monkeypatch.setattr(backend.webview, "start", lambda **kwargs: events.append("closed"))
    execute = Mock(side_effect=lambda *args: events.append("restart"))
    monkeypatch.setattr(os, "execv", execute)
    pipeline = SimpleNamespace(path=tmp_path, name="Pipeline")
    backend.start(pipeline, debug=True)
    if restart:
        assert events == ["closed", "restart"]
        execute.assert_called_once_with(sys.executable, [
            sys.executable, "-m", "collectra.main", "--pipeline",
            str(tmp_path / "pipeline.yaml"), "gui", "--debug",
        ])
    else:
        execute.assert_not_called()
