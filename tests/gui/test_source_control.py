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


def test_status_add_and_commit_from_nested_folder(repository):
    root, folder = repository
    api = GUIBackend(SimpleNamespace())
    api._yaml_path = str(folder / "results.yaml")
    (folder / "results.yaml").write_text("changed")
    (root / "outside-results.txt").write_text("also staged")
    status = api.git_action("status")
    assert status["success"]
    assert status["repository"] == str(root.resolve())
    assert "modified:" in status["output"]
    assert api.git_action("add", repository=status["repository"])["success"]
    assert "outside-results.txt" in git(root, "diff", "--cached", "--name-only")
    message = "A quoted message; $(touch should-not-exist)\n\nMore details"
    assert api.git_action("commit", message, status["repository"])["success"]
    assert git(root, "log", "-1", "--format=%B").strip() == message
    assert not (root / "should-not-exist").exists()
    assert "nothing to commit" in api.git_action("status")["output"]


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
    assert run_git(folder, "pull")["success"]
    assert (root / "remote-change.txt").read_text() == "remote"
    (folder / "results.yaml").write_text("local")
    assert run_git(folder, "add")["success"]
    assert run_git(folder, "commit", "Local change")["success"]
    assert run_git(folder, "push")["success"]
    assert git(root, "rev-parse", "HEAD") == git(
        remote, "rev-parse", f"refs/heads/{branch}"
    )


def test_errors_and_changed_repository(repository, tmp_path):
    root, folder = repository
    assert not run_git(tmp_path, "status")["success"]
    assert not run_git(folder, "reset")["success"]
    assert not run_git(folder, "commit", "  ")["success"]
    result = run_git(folder, "add", repository=str(tmp_path / "different"))
    assert not result["success"]
    assert "changed" in result["error"]
    failure = run_git(folder, "commit", "No changes")
    assert not failure["success"]
    assert "nothing to commit" in failure["error"]


def test_backend_requires_open_results_and_serializes_operations(repository):
    root, folder = repository
    api = GUIBackend(SimpleNamespace())
    assert not api.git_action("status")["success"]
    api._yaml_path = str(folder / "results.yaml")
    api._git_lock.acquire()
    try:
        assert "already running" in api.git_action("add")["error"]
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
