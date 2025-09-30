from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass(kw_only=True)
class MetaClass(ABC):
    
    @abstractmethod
    def metadata(self) -> dict:
        pass