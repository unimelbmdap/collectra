from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import Artefact

__all__ = ["Link"]


@dataclass
class Link(Artefact):
    """A soft reference to the artefact identified by its first parent."""

    target: Artefact | None = field(default=None, repr=False, compare=False)

    @property
    def parent_id(self) -> str:
        return self.parents[0] if self.parents else ""

    @property
    def __class__(self) -> type:
        """Present the resolved target type to ``isinstance`` consumers."""
        target = self.__dict__.get("target")
        return target.__class__ if target is not None else Link

    def bind(self, artefacts: dict[str, Artefact]) -> Artefact | None:
        """Resolve the first parent ID and retain it as the runtime target."""
        self.target = artefacts.get(self.parent_id)
        return self.target

    def resolve(self) -> Artefact:
        if self.target is None:
            raise RuntimeError(
                f"Link {self.id!r} has not been resolved to parent {self.parent_id!r}"
            )
        return self.target

    def __call__(self) -> Any:
        return self.resolve()()

    def __getattr__(self, name: str) -> Any:
        # Dataclass initialization probes attributes before ``target`` exists.
        target = self.__dict__.get("target")
        if target is None:
            raise AttributeError(name)
        return getattr(target, name)

    def attributes_to_ignore(self) -> set:
        attributes = super().attributes_to_ignore()
        attributes.add("target")
        return attributes

    def serialize(self) -> dict:
        """Serialize only the soft-link identity and parent reference."""
        serialized = {
            "type": self.get_class_path(),
            "id": self.id,
        }
        if self.parents:
            serialized["parents"] = (
                self.parents[0] if len(self.parents) == 1 else self.parents
            )
        return serialized

    def extract(self, path: Path | str) -> None:
        self.resolve().extract(path)

    def evaluate(self, gold: Artefact) -> float:
        if isinstance(gold, Link):
            if self.target is not None and gold.target is not None:
                return self.target.evaluate(gold.target)
            return float(bool(self.parent_id) and self.parent_id == gold.parent_id)
        return self.resolve().evaluate(gold)
