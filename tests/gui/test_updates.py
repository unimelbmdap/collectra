from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from collectra.gui.backend import GUIBackend
from collectra.gui.updates import collectra_repository, git, inspect_repository, update_repository


@pytest.fixture
def repositories(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    writer = tmp_path / "writer"
    subprocess.run(["git", "clone", str(remote), str(writer)], check=True, capture_output=True)
    git(writer, "config", "user.name", "Test")
    git(writer, "config", "user.email", "test@example.com")
    (writer / "file.txt").write_text("first")
    git(writer, "add", ".")
    git(writer, "commit", "-m", "Initial")
    git(writer, "push", "-u", "origin", "HEAD")
    reader = tmp_path / "reader"
    subprocess.run(["git", "clone", str(remote), str(reader)], check=True, capture_output=True)
    git(reader, "config", "user.name", "Test")
    git(reader, "config", "user.email", "test@example.com")
    return writer, reader


def advance(writer):
    (writer / "file.txt").write_text("latest")
    git(writer, "add", ".")
    git(writer, "commit", "-m", "Update")
    git(writer, "push")


def test_check_fetches_without_changing_files_and_update_fast_forwards(repositories):
    writer, reader = repositories
    assert inspect_repository(reader, "Pipeline")["status"] == "current"
    advance(writer)
    result = inspect_repository(reader, "Pipeline")
    assert result["can_update"]
    assert result["behind"] == 1
    assert (reader / "file.txt").read_text() == "first"
    assert update_repository(reader, "Pipeline")["status"] == "updated"
    assert (reader / "file.txt").read_text() == "latest"


def test_update_rechecks_local_changes(repositories):
    writer, reader = repositories
    advance(writer)
    assert inspect_repository(reader, "Collectra")["can_update"]
    (reader / "file.txt").write_text("local edits")
    result = update_repository(reader, "Collectra")
    assert result["status"] == "blocked"
    assert not result["can_update"]
    assert (reader / "file.txt").read_text() == "local edits"


def test_diverged_branch_is_not_merged(repositories):
    writer, reader = repositories
    advance(writer)
    (reader / "local.txt").write_text("local")
    git(reader, "add", ".")
    git(reader, "commit", "-m", "Local commit")
    before = git(reader, "rev-parse", "HEAD")
    assert update_repository(reader, "Pipeline")["status"] == "blocked"
    assert git(reader, "rev-parse", "HEAD") == before


def test_non_repo_and_missing_remote(tmp_path, repositories):
    assert inspect_repository(tmp_path, "Pipeline")["status"] == "unavailable"
    _, reader = repositories
    git(reader, "remote", "remove", "origin")
    assert inspect_repository(reader, "Collectra")["message"] == "No Git remote configured."


def test_detached_and_untracked_branches_are_not_updated(repositories):
    _, reader = repositories
    git(reader, "checkout", "--detach")
    assert not inspect_repository(reader, "Pipeline")["can_update"]
    git(reader, "checkout", "-b", "untracked")
    assert not inspect_repository(reader, "Pipeline")["can_update"]


def test_fetch_failure_is_reported(repositories):
    _, reader = repositories
    git(reader, "remote", "set-url", "origin", str(reader / "missing.git"))
    result = inspect_repository(reader, "Pipeline")
    assert result["status"] == "error"
    assert not result["can_update"]


def test_collectra_checkout_must_track_imported_source(repositories, monkeypatch):
    import collectra

    _, reader = repositories
    source = reader / "collectra" / "__init__.py"
    source.parent.mkdir()
    source.write_text("")
    monkeypatch.setattr(collectra, "__file__", str(source))
    assert collectra_repository() is None
    git(reader, "add", ".")
    assert collectra_repository() == reader


def test_backend_checks_both_repositories_and_restricts_targets(tmp_path, monkeypatch):
    from collectra.gui import updates

    api = GUIBackend(SimpleNamespace(path=tmp_path, ext="collectra"))
    calls = []
    monkeypatch.setattr(updates, "collectra_repository", lambda: tmp_path / "collectra")
    def inspect(root, label):
        calls.append((root, label))
        return {"label": label, "status": "current", "can_update": False}
    monkeypatch.setattr(updates, "inspect_repository", inspect)
    assert [row["id"] for row in api.check_updates()["repositories"]] == ["pipeline", "collectra"]
    assert calls == [(tmp_path, "Pipeline"), (tmp_path / "collectra", "Collectra")]
    assert not api.install_update("arbitrary/path")["success"]
    api._update_lock.acquire()
    try:
        assert not api.check_updates()["success"]
        assert not api.install_update("pipeline")["success"]
    finally:
        api._update_lock.release()
