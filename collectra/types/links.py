from dataclasses import dataclass, field

from .base import Artefact


@dataclass
class Link(Artefact):
    target: Artefact
