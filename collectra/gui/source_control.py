"""Run the GUI's fixed Git operations in the active results repository."""

import os
import subprocess
from pathlib import Path


def run_git(
    directory: Path, action: str, message: str = "", repository: str = ""
) -> dict:
    commands = {
        "status": ["status"],
        "pull": ["pull", "--no-edit"],
        "add": ["add", "."],
        "commit": ["commit", "-m", message],
        "push": ["push"],
    }
    if action not in commands:
        return {"success": False, "error": "Unknown Git action"}
    if action == "commit" and not message.strip():
        return {"success": False, "error": "Enter a commit message"}
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
            return {
                "success": False,
                "error": root_result.stderr.strip()
                or "This folder is not in a Git repository",
            }
        root = Path(root_result.stdout.strip()).resolve()
        if repository and root != Path(repository).resolve():
            return {
                "success": False,
                "error": "The active repository changed. Reopen source control before continuing.",
            }
        result = subprocess.run(
            ["git", "-C", str(root), *commands[action]],
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
        )
        output = "\n".join(
            part.strip() for part in [result.stdout, result.stderr] if part.strip()
        )
        return {
            "success": result.returncode == 0,
            "repository": str(root),
            "action": action,
            "output": output or ("Files staged." if action == "add" else "Done."),
            "error": output if result.returncode else "",
        }
    except FileNotFoundError:
        return {"success": False, "error": "Git is not installed or cannot be found"}
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Git timed out. Check the repository status before retrying.",
        }
    except OSError as error:
        return {"success": False, "error": str(error)}
