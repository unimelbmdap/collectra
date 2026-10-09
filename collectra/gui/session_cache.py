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
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump({"folders": folders}, stream)
            os.replace(temporary, self.path)
        except OSError as error:
            logger.warning("Could not save GUI folder session: %s", error)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
