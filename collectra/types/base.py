from dataclasses import dataclass, field
from pathlib import Path
from collectra.commons import Node, NodeStatus


@dataclass
class Data(Node):

    data: str | Path = field(default="")

    def __post_init__(self):
        self._status = NodeStatus.READY if self.data else NodeStatus.NOT_READY

    def return_data(self) -> str:
        return str(self.data)

    def __call__(self) -> str:
        return self.return_data()
