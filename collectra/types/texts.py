from pathlib import Path
from dataclasses import dataclass

from .base import Data

__all__ = ["Text"]


@dataclass
class Text(Data):

    def __post_init__(self):        
        try:
            if self.data and Path(self.data).exists():
                self.data = Path(self.data)
        except OSError:
            pass

        if isinstance(self.data, Path) and self.data.exists() and self.data.is_file():            
            self.data = self.data.read_text()            
