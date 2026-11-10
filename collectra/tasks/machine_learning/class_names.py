"""The class names a trained model was given, saved in a file beside its checkpoint.

The file is YAML (``<checkpoint stem>.classes.yaml`` or ``classes.yaml``). The older
JSON files with the same names are still read.
"""

from pathlib import Path

import yaml

SUFFIXES = (".yaml", ".json")


def sidecar_candidates(checkpoint: Path) -> list[Path]:
    """Where the class names for a checkpoint may be, most specific first."""
    folder = Path(checkpoint).parent
    stems = (f"{Path(checkpoint).stem}.classes", "classes")
    return [folder / f"{stem}{suffix}" for stem in stems for suffix in SUFFIXES]


def read_class_names(checkpoint: Path) -> list[str] | None:
    """The class names saved beside ``checkpoint``, or None if there are none."""
    for path in sidecar_candidates(checkpoint):
        if path.exists():
            # JSON is YAML, so one reader serves both.
            data = yaml.safe_load(path.read_text()) or {}
            return data.get("classes")
    return None


def write_class_names(path: Path, classes: list[str]) -> None:
    """Save the class names, in order, as YAML."""
    path.write_text(yaml.safe_dump({"classes": list(classes)}, sort_keys=False))
