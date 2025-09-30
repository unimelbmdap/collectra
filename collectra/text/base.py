from dataclasses import dataclass
from pathlib import Path

@dataclass(kw_only=True)
class Text:

    content: str
    path: Path

    def __init__(self, content: str):
        self.content = content

    def __str__(self):
        return self.content