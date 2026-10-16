"""Run the GUI's fixed Git operations in the active results repository.

Everything returned to the page is worded for people who do not know Git.
"""

import os
import re
import subprocess
from pathlib import Path

ACTIONS = ("status", "pull", "save", "push")
MAX_LISTED_FILES = 50

NOT_LINKED = "This project is not linked to a shared copy yet."
CANNOT_REACH = "Could not reach the shared copy. Check your internet connection."
CANNOT_SIGN_IN = "Could not sign in to the shared copy."


def run_git(
    directory: Path, action: str, message: str = "", repository: str = ""
) -> dict:
    if action not in ACTIONS:
        return _failure("Unknown action")
    if action == "save" and not message.strip():
        return _failure("Enter a message first.")
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    environment.setdefault("GIT_SSH_COMMAND", "ssh -oBatchMode=yes")
    try:
        root_result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        )
        if root_result.returncode:
            return _failure("This folder is not set up for saving versions.")
        root = Path(root_result.stdout.strip()).resolve()
        if repository and root != Path(repository).resolve():
            return _failure(
                "The project changed. Close and reopen History and Changes before going on."
            )

        def git(*arguments):
            result = subprocess.run(
                ["git", "-C", str(root), *arguments],
                capture_output=True,
                text=True,
                timeout=120,
                env=environment,
            )
            output = "\n".join(
                part.strip() for part in [result.stdout, result.stderr] if part.strip()
            )
            return result.returncode, output

        if action == "status":
            code, output = git("status", "--porcelain=v1", "--branch", "-z")
            if code:
                return _failure(_plain_failure(output), root)
            return _success(_describe_status(output), root)
        if action == "save":
            code, output = git("add", ".")
            if code:
                return _failure(_plain_failure(output), root)
            code, output = git("commit", "-m", message)
            if code and "nothing to commit" in output:
                return _failure("There are no changes to save.", root)
            if code:
                return _failure(_plain_failure(output), root)
            return _success("Changes saved.", root)
        if action == "pull":
            code, output = git("pull", "--no-rebase", "--no-edit")
            if code and ("CONFLICT" in output or "Automatic merge failed" in output):
                git("merge", "--abort")
                return _failure(
                    "Your changes and the updates are in the same place, so nothing "
                    "was changed. Ask for help before going on.",
                    root,
                )
            if code:
                return _failure(_plain_failure(output), root)
            if "Already up to date" in output:
                return _success("Already up to date.", root)
            return _success("Updates received.", root)
        code, output = git("push")
        if code:
            return _failure(_plain_failure(output), root)
        if "up-to-date" in output or "up to date" in output:
            return _success("Nothing new to share.", root)
        return _success("Changes shared.", root)
    except FileNotFoundError:
        return _failure(
            "Saving versions needs Git, which is not installed on this computer."
        )
    except subprocess.TimeoutExpired:
        return _failure(
            "That took too long. Check what has changed before trying again."
        )
    except OSError as error:
        return _failure(str(error))


def _success(text: str, root: Path) -> dict:
    return {"success": True, "repository": str(root), "output": text, "error": ""}


def _failure(text: str, root: Path | None = None) -> dict:
    result = {"success": False, "output": text, "error": text}
    if root is not None:
        result["repository"] = str(root)
    return result


def _plain_failure(output: str) -> str:
    """Git's own message turned into plain words where we know the cause."""
    if "would be overwritten" in output:
        return "Save your changes first, then get updates."
    if "rejected" in output or "fetch first" in output:
        return "Others have shared changes. Get updates first."
    if "no tracking information" in output or "no upstream" in output:
        return NOT_LINKED
    if (
        "No configured push destination" in output
        or "does not appear to be a git repository" in output
    ):
        return NOT_LINKED
    if "Could not resolve host" in output or "unable to access" in output:
        return CANNOT_REACH
    if re.search(
        r"Authentication failed|could not read Username|Permission denied", output
    ):
        return CANNOT_SIGN_IN
    return output or "Something went wrong."


def _describe_status(porcelain: str) -> str:
    entries = porcelain.split("\0")
    branch = entries[0] if entries and entries[0].startswith("## ") else ""
    changes = {"Edited": [], "New": [], "Deleted": []}
    skip_next = False
    for entry in entries[1:]:
        if skip_next:  # the old name of a renamed file
            skip_next = False
            continue
        if len(entry) < 4:
            continue
        codes, path = entry[:2], entry[3:]
        skip_next = codes[0] in "RC"
        if codes == "??" or "A" in codes:
            changes["New"].append(path)
        elif "D" in codes:
            changes["Deleted"].append(path)
        else:
            changes["Edited"].append(path)
    lines = []
    listed = [(kind, path) for kind, paths in changes.items() for path in paths]
    if listed:
        lines.append("Not saved yet:")
        lines.extend(f"  {kind}: {path}" for kind, path in listed[:MAX_LISTED_FILES])
        if len(listed) > MAX_LISTED_FILES:
            lines.append(f"  ...and {len(listed) - MAX_LISTED_FILES} more")
    ahead = re.search(r"ahead (\d+)", branch)
    behind = re.search(r"behind (\d+)", branch)
    notes = []
    if ahead:
        count = int(ahead.group(1))
        notes.append(
            f"{count} saved version has not been shared yet."
            if count == 1
            else f"{count} saved versions have not been shared yet."
        )
    if behind:
        count = int(behind.group(1))
        notes.append(
            "1 update is waiting to be received."
            if count == 1
            else f"{count} updates are waiting to be received."
        )
    if lines and notes:
        lines.append("")
    lines.extend(notes)
    return "\n".join(lines) or "Everything is saved and up to date."
