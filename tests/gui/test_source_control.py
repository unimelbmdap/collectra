"""Exercise GUI source control against local repositories and a local remote."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from collectra.gui.backend import GUIBackend
from collectra.gui.source_control import run_git


def git(directory, *args):
    return subprocess.run(
        ["git", "-C", str(directory), *args], capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    folder = root / "results" / "page.collectra"
    folder.mkdir(parents=True)
    (folder / "results.yaml").write_text("initial")
    git(root, "add", ".")
    git(root, "commit", "-m", "Initial")
    return root, folder


def test_status_and_save_from_nested_folder(repository):
    root, folder = repository
    api = GUIBackend(SimpleNamespace())
    api._yaml_path = str(folder / "results.yaml")
    (folder / "results.yaml").write_text("changed")
    (root / "outside-results.txt").write_text("also saved")
    status = api.git_action("status")
    assert status["success"]
    assert status["repository"] == str(root.resolve())
    assert "Not saved yet:" in status["output"]
    assert "Edited: results/page.collectra/results.yaml" in status["output"]
    assert "New: outside-results.txt" in status["output"]
    message = "A quoted message; $(touch should-not-exist)\n\nMore details"
    saved = api.git_action("save", message, status["repository"])
    assert saved["success"]
    assert saved["output"] == "Changes saved."
    assert "outside-results.txt" in git(
        root, "show", "--name-only", "--format=", "HEAD"
    )
    assert git(root, "log", "-1", "--format=%B").strip() == message
    assert not (root / "should-not-exist").exists()
    assert api.git_action("status")["output"] == "Everything is saved and up to date."


def test_status_lists_deleted_files(repository):
    root, folder = repository
    (folder / "results.yaml").unlink()
    assert (
        "Deleted: results/page.collectra/results.yaml"
        in run_git(folder, "status")["output"]
    )


def test_push_and_pull_with_local_remote(repository, tmp_path):
    root, folder = repository
    remote = tmp_path / "remote.git"
    remote.mkdir()
    git(remote, "init", "--bare")
    git(root, "remote", "add", "origin", str(remote))
    branch = git(root, "branch", "--show-current").strip()
    git(root, "push", "-u", "origin", branch)
    other = tmp_path / "other"
    subprocess.run(
        ["git", "clone", "--branch", branch, str(remote), str(other)],
        check=True,
        capture_output=True,
    )
    git(other, "config", "user.email", "test@example.com")
    git(other, "config", "user.name", "Test")
    (other / "remote-change.txt").write_text("remote")
    git(other, "add", ".")
    git(other, "commit", "-m", "Remote change")
    git(other, "push")
    status = run_git(folder, "status")
    assert status["output"] == "Everything is saved and up to date."
    git(root, "fetch")
    assert run_git(folder, "status")["output"] == "1 update is waiting to be received."
    pulled = run_git(folder, "pull")
    assert pulled["success"]
    assert pulled["output"] == "Updates received."
    assert run_git(folder, "pull")["output"] == "Already up to date."
    assert (root / "remote-change.txt").read_text() == "remote"
    (folder / "results.yaml").write_text("local")
    assert run_git(folder, "save", "Local change")["success"]
    assert (
        run_git(folder, "status")["output"]
        == "1 saved version has not been shared yet."
    )
    shared = run_git(folder, "push")
    assert shared["success"]
    assert shared["output"] == "Changes shared."
    assert run_git(folder, "push")["output"] == "Nothing new to share."
    assert git(root, "rev-parse", "HEAD") == git(
        remote, "rev-parse", f"refs/heads/{branch}"
    )


def test_errors_are_in_plain_words(repository, tmp_path):
    root, folder = repository
    outside = run_git(tmp_path, "status")
    assert not outside["success"]
    assert outside["error"] == "This folder is not set up for saving versions."
    for old_name in ("add", "commit", "reset"):
        assert not run_git(folder, old_name)["success"]
    blank = run_git(folder, "save", "  ")
    assert not blank["success"]
    assert blank["error"] == "Enter a message first."
    result = run_git(folder, "status", repository=str(tmp_path / "different"))
    assert not result["success"]
    assert "changed" in result["error"]
    nothing = run_git(folder, "save", "No changes")
    assert not nothing["success"]
    assert nothing["error"] == "There are no changes to save."


def test_pull_clash_is_explained(repository, tmp_path):
    root, folder = repository
    remote = tmp_path / "remote.git"
    remote.mkdir()
    git(remote, "init", "--bare")
    git(root, "remote", "add", "origin", str(remote))
    branch = git(root, "branch", "--show-current").strip()
    git(root, "push", "-u", "origin", branch)
    other = tmp_path / "other"
    subprocess.run(
        ["git", "clone", "--branch", branch, str(remote), str(other)],
        check=True,
        capture_output=True,
    )
    git(other, "config", "user.email", "test@example.com")
    git(other, "config", "user.name", "Test")
    (other / "results/page.collectra/results.yaml").write_text("theirs")
    git(other, "commit", "-am", "Theirs")
    git(other, "push")
    (folder / "results.yaml").write_text("mine")
    git(root, "commit", "-am", "Mine")
    result = run_git(folder, "pull")
    assert not result["success"]
    assert result["error"] == (
        "Your changes and the updates are in the same place, so nothing was changed. "
        "Ask for help before going on."
    )
    assert (folder / "results.yaml").read_text() == "mine"
    assert not (root / ".git" / "MERGE_HEAD").exists()


def test_pull_merges_when_both_sides_saved_different_files(repository, tmp_path):
    root, folder = repository
    remote = tmp_path / "remote.git"
    remote.mkdir()
    git(remote, "init", "--bare")
    git(root, "remote", "add", "origin", str(remote))
    branch = git(root, "branch", "--show-current").strip()
    git(root, "push", "-u", "origin", branch)
    other = tmp_path / "other"
    subprocess.run(
        ["git", "clone", "--branch", branch, str(remote), str(other)],
        check=True,
        capture_output=True,
    )
    git(other, "config", "user.email", "test@example.com")
    git(other, "config", "user.name", "Test")
    (other / "theirs.txt").write_text("theirs")
    git(other, "add", ".")
    git(other, "commit", "-m", "Theirs")
    git(other, "push")
    (root / "mine.txt").write_text("mine")
    git(root, "add", ".")
    git(root, "commit", "-m", "Mine")
    result = run_git(folder, "pull")
    assert result["success"]
    assert result["output"] == "Updates received."
    assert (root / "theirs.txt").exists() and (root / "mine.txt").exists()


def test_backend_requires_open_results_and_serializes_operations(repository):
    root, folder = repository
    api = GUIBackend(SimpleNamespace())
    assert not api.git_action("status")["success"]
    api._yaml_path = str(folder / "results.yaml")
    api._git_lock.acquire()
    try:
        assert "already running" in api.git_action("status")["error"]
    finally:
        api._git_lock.release()


def test_linked_worktree_repository(repository, tmp_path):
    root, folder = repository
    worktree = tmp_path / "worktree"
    git(root, "worktree", "add", "-b", "worktree-test", str(worktree))
    assert (worktree / ".git").is_file()
    result = run_git(worktree / "results/page.collectra", "status")
    assert result["success"]
    assert result["repository"] == str(worktree.resolve())
