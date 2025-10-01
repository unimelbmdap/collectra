from pathlib import Path

from .base import Type


class Text(Type, str):
    def __new__(cls, data:str|Path) -> str:
        if Path(data).exists():
            data = Path(data)
        
        if isinstance(data, Path) and data.exists() and data.is_file():
            data = data.read_text()

        return super().__new__(cls, data)

