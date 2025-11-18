import enum

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, Generic

__all__ = ["T", "BaseEntity", "Node", "NodeStatus"]

T = TypeVar("T")


class BaseEntity(ABC, Generic[T]):

    name: str

    def __str__(self) -> str:
        return f"{self.name}\n{self.get_class_path()}"

    @abstractmethod
    def __call__(self) -> T:
        pass

    def get_name(self) -> str:
        """Get the name of the task.

        Returns:
            str: The name of the task.
        """
        return self.name

    def serialize(self) -> dict:
        serialized = dict(type=self.get_class_path())        
        for key, value in self.attributes.items():
            if isinstance(value, Path):
                value = value.name
            serialized[key] = value
        return serialized

    @property
    def attributes(self) -> dict:
        ignore = self.attributes_to_ignore()
        return {k: v for k, v in self.__dict__.items() if k not in ignore}

    def attributes_to_ignore(self) -> set:
        return set()

    @classmethod
    def get_class_path(cls: type) -> str:
        """Return the fully qualified path for a class.

        Example:
            >>> from collectra.images import Image
            >>> get_class_path(Image)
            'collectra.images.Image'
        """
        return f"{cls.__module__.split('.')[0]}.{cls.__name__}"


class NodeStatus(enum.Enum):
    NOT_READY = "not_ready"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value


@dataclass
class Node:

    name: str
    status: NodeStatus = field(init=False, default=NodeStatus.NOT_READY)

    def process(self) -> list:
        return []
