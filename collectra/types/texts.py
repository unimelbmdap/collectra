__all__ = ["Text"]

from pathlib import Path
from dataclasses import dataclass

from .nodes import Node

@dataclass(kw_only=True)
class Text(Node, str):

    def __new__(cls, data: str | Path, name: str = "") -> str:
        try:
            if Path(data).exists():
                data = Path(data)
        except OSError:
            pass

        if isinstance(data, Path) and data.exists() and data.is_file():
            data = data.read_text()

        return super().__new__(cls, data)


    def __init__(self, data: str | Path, name: str = ""):
        # Now initialize Node
        if name is None:
            name = str(data)[:20]  # or whatever default makes sense
        Node.__init__(self, name)