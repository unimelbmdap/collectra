"""Check and fast-forward the two source repositories used by this GUI."""

import os
import subprocess
from pathlib import Path


def git(root, *args):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    env.setdefault("GIT_SSH_COMMAND", "ssh -oBatchMode=yes")
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        timeout=120, env=env,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Git command failed")
    return result.stdout.strip()


def collectra_repository():
    import collectra

    source = Path(collectra.__file__).resolve()
    for root in source.parents:
        if (root / ".git").exists():
            try:
                git(root, "ls-files", "--error-unmatch", "--", str(source.relative_to(root)))
                return root
            except (RuntimeError, OSError, subprocess.TimeoutExpired):
                return None
    return None


def inspect_repository(root, label, fetch=True):
    result = {"label": label, "path": str(root) if root else "", "can_update": False}
    if root is None or not (Path(root) / ".git").exists():
        return {**result, "status": "unavailable", "message": "Not installed from a Git checkout."}
    root = Path(root)
    try:
        branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
        if branch == "HEAD":
            raise RuntimeError("This checkout has a detached HEAD. Switch to a tracking branch before updating.")
        remotes = git(root, "remote").splitlines()
        if not remotes:
            return {**result, "status": "unavailable", "message": "No Git remote configured."}
        remote = git(root, "for-each-ref", "--format=%(upstream:remotename)", f"refs/heads/{branch}")
        if remote not in remotes:
            raise RuntimeError("The current branch does not track a remote branch.")
        if fetch:
            git(root, "fetch", "--prune", "--", remote)
        upstream = git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
        ahead, behind = map(int, git(root, "rev-list", "--left-right", "--count", "HEAD...@{upstream}").split())
        dirty = bool(git(root, "status", "--porcelain"))
        result.update(branch=branch, upstream=upstream, ahead=ahead, behind=behind)
        if dirty:
            return {**result, "status": "blocked", "message": "Local changes are present. Commit or stash them before updating."}
        if ahead and behind:
            return {**result, "status": "blocked", "message": "Local and remote branches have diverged. Resolve this in Git before updating."}
        if behind:
            return {**result, "status": "available", "can_update": True,
                    "message": f"{behind} new commit(s) available from {upstream}."}
        return {**result, "status": "current", "message": "Up to date." if not ahead else "No incoming updates; local commits are ahead of the remote."}
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        return {**result, "status": "error", "message": str(error)}


def update_repository(root, label):
    # Recheck immediately before modifying a checkout: status may have changed
    # since the dialog opened. Never merge or reset local work.
    result = inspect_repository(root, label)
    if not result["can_update"]:
        return result
    try:
        git(root, "merge", "--ff-only", "@{upstream}")
        return {**result, "can_update": False, "status": "updated",
                "message": "Updated successfully. Restart the GUI to use the new code."}
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        return {**result, "can_update": False, "status": "error", "message": str(error)}
