from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from collectra.commons import BaseEntity, Node, NodeStatus

__all__ = ["Data", "DataNode"]

@dataclass
class Data(BaseEntity):

    name: str
    data: str | Path = field(default="")

    def return_data(self) -> str:
        return str(self.data)

    def __call__(self) -> str:
        return self.return_data()

    @abstractmethod
    def handle(self) -> None:
        """Handle data preparation or loading logic."""
        pass

@dataclass
class DataNode(Node):

    items: list[Data] = field(default_factory=list)
    types: list[type] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._status = NodeStatus.READY if self.items else NodeStatus.NOT_READY
    
    def add_item(self, item: Data) -> None:
        self.items.append(item)
        self._status = NodeStatus.READY
    
    def add_type(self, type_: type) -> None:
        self.types.append(type_)
    
    def check_type(self, type_: type) -> bool:
        return type_ in self.types

    def __str__(self) -> str:        
        types_str = "\n".join([t.get_class_path() for t in self.types])
        return f"{self.name}\n{types_str}"