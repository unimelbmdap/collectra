import enum
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

from rich.table import Table

__all__ = ["T", "BaseEntity", "Node", "NodeStatus"]

T = TypeVar("T")


class ErrorType(enum.Enum):
    """Enumeration of error types."""

    WARNING = "warning"
    ERROR = "error"


class ErrorEntity:
    """Represents an error that occurred during processing."""

    def __init__(self, message: str, type: ErrorType) -> None:
        self.type: ErrorType = type
        self.message: str = message
        self.traceback: str = traceback.format_exc() if type == ErrorType.ERROR else ""

    def __str__(self) -> str:
        return f"{self.type}: {self.message}"


class ErrorManager:

    def __init__(self) -> None:
        self.errors: dict[ErrorType, list[ErrorEntity]] = {}

    def _add_message(self, message: str, type: ErrorType = ErrorType.ERROR) -> None:
        if type not in self.errors:
            self.errors[type] = []
        self.errors[type].append(ErrorEntity(message, type))

    def flush_msg(self, logger: Table):
        if not self.errors:
            return
        for err_type, messages in self.errors.items():
            for msg in messages:
                logger.add_row(str(err_type), msg.message, msg.traceback)

    def set_err(self, message: str) -> None:
        self._add_message(message, ErrorType.ERROR)

    def set_warn(self, message: str) -> None:
        self._add_message(message, ErrorType.WARNING)


class BaseEntity(ABC, Generic[T]):

    def __init__(self, name: str) -> None:
        self.name = name
        self.catcher = ErrorManager()

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
        return set(["catcher"])

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

    def __post_init__(self) -> None:
        self.catcher = ErrorManager()

    def process(self) -> list:
        return []
