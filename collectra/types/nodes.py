import enum

class NodeStatus(enum.Enum):
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Node:
    def __init__(self, name: str):
        self.name = name
        self.status = NodeStatus.READY

    def set_status(self, status: NodeStatus):
        self.status = status

    def get_status(self) -> NodeStatus:
        return self.status