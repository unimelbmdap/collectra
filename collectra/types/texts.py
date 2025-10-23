from pathlib import Path
from dataclasses import dataclass, field

from .base import Data

__all__ = ["Text"]


@dataclass
class Text(Data):

    data: str | Path = field(default="")

    def __post_init__(self):
        try:
            if self.data and Path(self.data).exists():
                self.data = Path(self.data)
        except OSError:
            pass

        if isinstance(self.data, Path) and self.data.exists() and self.data.is_file():
            self.data = self.data.read_text()

    def __call__(self) -> str | Path:
        return self.data
