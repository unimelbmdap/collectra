from dataclasses import dataclass, field

from .base import Data

@dataclass
class Link(Data):
    target:Data

    