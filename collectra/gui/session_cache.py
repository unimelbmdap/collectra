"""Best-effort, per-user storage of each pipeline's open GUI folders."""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)


class FolderSessionCache:
    def __init__(self, pipeline_directory: Path):
        key = hashlib.sha256(
            str(Path(pipeline_directory).resolve()).encode()
        ).hexdigest()
        self.path = (
            Path(user_cache_dir("collectra", appauthor=False))
            / "gui-sessions"
            / f"{key}.json"
        )

        self.active_path = self.path.with_name(f"{key}-active.json")

    def read_active(self) -> str | None:
        try:
            active = json.loads(self.active_path.read_text(encoding="utf-8"))
            return active if isinstance(active, str) else None
        except (OSError, ValueError):
            return None

    def write_active(self, folder: str) -> None:
        self._write_json(self.active_path, folder)

    def read(self) -> list[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            folders = data.get("folders") if isinstance(data, dict) else None
            if isinstance(folders, list) and all(
                isinstance(path, str) for path in folders
            ):
                return folders
        except (OSError, ValueError):
            pass
        return []

    def write(self, folders: list[str]) -> None:
        self._write_json(self.path, {"folders": folders})

    def _write_json(self, destination: Path, data) -> None:
        temporary = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream)
            os.replace(temporary, destination)
        except OSError as error:
            logger.warning("Could not save GUI folder session: %s", error)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
