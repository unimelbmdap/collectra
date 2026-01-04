from dataclasses import dataclass, field
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path

from markdown import Markdown as MDown

from .base import Data

__all__ = ["Text"]


def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


# patching Markdown
MDown.output_formats["plain"] = unmark_element
__md = MDown(output_format="plain")
__md.stripTopLevelTags = False


def unmark(text):
    return __md.convert(text)


@dataclass
class Text(Data):

    data: str | Path = field(default="")

    def __post_init__(self):
        super().__post_init__()
        try:
            if self.data and Path(self.data).exists():
                self.data = Path(self.data)
        except OSError:
            pass

        if isinstance(self.data, Path) and self.data.exists() and self.data.is_file():
            self.data = self.data.read_text()

    def __call__(self) -> str | Path:
        return self.data

    def eval(self, gold: "Text") -> dict:
        if not isinstance(gold, Text):
            raise ValueError("Reference data must be an instance of Text.")
        ratio = SequenceMatcher(None, str(self.data), str(gold.data)).ratio()
        return ratio
