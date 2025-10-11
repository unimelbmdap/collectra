from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass(kw_only=True)
class MetaClass(ABC):
    def __init__(self, **kwargs) -> str:
        return super().__init__()

    @abstractmethod
    def metadata(self) -> dict:
        pass
