"""Plan which tasks a run of one page will run, re-run or skip."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TaskPlan:
    """What a run does with one task, and why."""

    name: str
    action: Literal["run", "rerun", "skip"]
    reason: str


@dataclass(frozen=True)
class RunPlan:
    """The plan for every task of a pipeline, in pipeline order."""

    tasks: list[TaskPlan]

    @property
    def to_run(self) -> list[str]:
        """Names of the tasks that will run or re-run."""
        return [task.name for task in self.tasks if task.action != "skip"]


def plan_tasks(
    missing: set[str], children: dict[str, set[str]], order: list[str]
) -> RunPlan:
    """Plan a run given the tasks that have no output yet.

    A missing task runs. A task downstream of a running task, however far,
    re-runs because its input changes; its reason names the nearest running
    task upstream. Every other task is skipped.
    """
    upstream: dict[str, str] = {}
    queue = deque((name, name) for name in order if name in missing)
    while queue:
        name, source = queue.popleft()
        for child in sorted(children.get(name, ())):
            if child not in missing and child not in upstream:
                upstream[child] = source
                queue.append((child, child))
    plans = []
    for name in order:
        if name in missing:
            plans.append(TaskPlan(name, "run", "no output yet"))
        elif name in upstream:
            plans.append(TaskPlan(name, "rerun", f"upstream {upstream[name]} runs"))
        else:
            plans.append(TaskPlan(name, "skip", "already has output"))
    return RunPlan(plans)
