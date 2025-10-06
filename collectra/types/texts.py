from pathlib import Path

from .base import Type


class Text(Type, str):
    def __new__(cls, data:str|Path) -> str:
        try:
            if Path(data).exists():
                data = Path(data)
        except OSError:
            pass
        
        if isinstance(data, Path) and data.exists() and data.is_file():
            data = data.read_text()

        return super().__new__(cls, data)

