"""Best-effort, per-user storage of the GUI's panel and tab arrangement.

Lives in the user's own cache directory (never in the shared project
folder), so one person's layout cannot overwrite another's.
"""

import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path

import yaml
from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)

VERSION = 1
TAB_KINDS = {"dag", "table", "node"}
DIRECTIONS = {"left", "right", "above", "below"}
MAX_LAYOUT_OPS = 200
PIXEL_FLEX = re.compile(r"^0 [01] \d+(\.\d+)?(px|%)$")


def sanitize_arrangement(data):
    """Keep only the parts of a saved arrangement the GUI understands.

    Returns None when ``data`` is not a usable arrangement.
    """
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return None
    result = {"version": VERSION}
    if isinstance(data.get("focused"), str):
        result["focused"] = data["focused"]
    panels = {}
    raw_panels = data.get("panels")
    for name, panel in (raw_panels.items() if isinstance(raw_panels, dict) else []):
        if not isinstance(name, str) or not isinstance(panel, dict):
            continue
        tabs = [tab for tab in map(_sanitize_tab, panel.get("tabs") or []) if tab]
        active = panel.get("active")
        if type(active) is not int or not 0 <= active < len(tabs):
            active = 0
        panels[name] = {"active": active, "tabs": tabs}
    result["panels"] = panels
    # Absent in saves from before this was remembered: then nothing is changed on restore.
    if isinstance(data.get("collapsed"), list):
        result["collapsed"] = [
            name for name in data["collapsed"] if isinstance(name, str) and name
        ]
    result["layout"] = _sanitize_layout(data.get("layout"))
    result["sizes"] = _sanitize_sizes(data.get("sizes"))
    return result


def _sanitize_layout(ops):
    """The log of panel splits and closes, replayed in order on restore."""
    clean = []
    for op in ops if isinstance(ops, list) else []:
        if not isinstance(op, dict) or not isinstance(op.get("key"), str):
            continue
        if op.get("op") == "split":
            if isinstance(op.get("from"), str) and op.get("direction") in DIRECTIONS:
                clean.append(
                    {
                        "op": "split",
                        "from": op["from"],
                        "direction": op["direction"],
                        "key": op["key"],
                    }
                )
        elif op.get("op") == "remove":
            clean.append({"op": "remove", "key": op["key"]})
    return clean[-MAX_LAYOUT_OPS:]


def _sanitize_sizes(sizes):
    return {
        name: size
        for name, size in (sizes.items() if isinstance(sizes, dict) else [])
        if isinstance(name, str) and isinstance(size, str) and PIXEL_FLEX.match(size)
    }


def _sanitize_tab(tab):
    if not isinstance(tab, dict) or tab.get("kind") not in TAB_KINDS:
        return None
    if tab["kind"] in ("dag", "table"):
        return {"kind": tab["kind"]}
    group = tab.get("group")
    if not isinstance(group, str) or not group:
        return None
    return {"kind": "node", "group": group}


class ArrangementStore:
    def __init__(
        self,
        pipeline_directory: Path,
        pipeline_name: str | None = None,
        default_layout: Path | None = None,
    ):
        key = hashlib.sha256(
            str(Path(pipeline_directory).resolve()).encode()
        ).hexdigest()
        directory = (
            Path(user_cache_dir("collectra", appauthor=False))
            / "gui-arrangements"
        )
        name = pipeline_name or Path(pipeline_directory).name
        name = re.sub(r'[/\\\x00-\x1f]', "_", name).strip()
        if name in {"", ".", ".."}:
            name = "pipeline"
        self.path = directory / name / f"{key}.yaml"
        self.legacy_path = directory / f"{key}.yaml"
        self.default_layout = default_layout

    def read(self):
        for path in (self.path, self.legacy_path, self.default_layout):
            if path is not None:
                arrangement = self._read_file(path)
                if arrangement is not None:
                    return arrangement
        return None

    @staticmethod
    def _read_file(path):
        try:
            return sanitize_arrangement(
                yaml.safe_load(path.read_text(encoding="utf-8"))
            )
        except (OSError, UnicodeError, yaml.YAMLError):
            return None

    def custom_layouts(self):
        try:
            return sorted(
                (p for p in self.path.parent.glob("*.y*ml")
                 if p != self.path and not re.fullmatch(r"[0-9a-f]{64}", p.stem)
                 and p.suffix in {".yaml", ".yml"}),
                key=lambda p: p.name.casefold(),
            )
        except OSError:
            return []

    def save_named(self, name: str, arrangement: dict):
        if (not isinstance(name, str) or not name.strip()
                or name.strip() in {".", ".."}
                or re.search(r'[/\\\x00-\x1f]', name)
                or re.fullmatch(r"[0-9a-f]{64}", name.strip())):
            raise ValueError("Enter a layout name without slashes or control characters.")
        path = self.path.parent / (name.strip() + ".yaml")
        if path.exists():
            raise ValueError("A layout with that name already exists. Choose another name.")
        self._write_file(path, arrangement, raise_errors=True)

    def write(self, arrangement: dict) -> None:
        self._write_file(self.path, arrangement)

    @staticmethod
    def _write_file(path, arrangement, raise_errors=False):
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                yaml.safe_dump(arrangement, stream, sort_keys=False)
            os.replace(temporary, path)
        except OSError as error:
            logger.warning("Could not save GUI arrangement: %s", error)
            if raise_errors:
                raise
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
