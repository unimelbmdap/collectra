"""The GUI graph says whether a node is a task or an artefact, and what artefacts hold."""

from collectra import Pipeline
from collectra.tasks.base import Task
from collectra.types.images import Image
from collectra.types.links import Link
from collectra.types.texts import Text

THIS = "tests.unit.test_gui_node_types"


class Reads(Task):
    def run(self, image: Image) -> Text:
        return Text(name=self.get_output_name(), data="x")


class Sorts(Task):
    def run(self, image: Image) -> Link:
        return Link(name=self.get_output_name(), parents=[image.id])


def graph(tmp_path, **tasks):
    pipeline = Pipeline("Nodes", "nds", "0.1.0", path=str(tmp_path), **tasks)
    return {node["id"]: node for node in pipeline.gui_graph()["nodes"]}


def test_tasks_are_tasks_and_artefacts_are_artefacts(tmp_path):
    nodes = graph(
        tmp_path,
        read={"type": f"{THIS}.Reads", "input": ["page"], "output": ["text"]},
    )
    assert nodes["read"]["nodeType"] == "Task"
    assert nodes["page"]["nodeType"] == "Artefact"
    assert nodes["text"]["nodeType"] == "Artefact"


def test_what_a_task_is_does_not_depend_on_its_class_name(tmp_path):
    nodes = graph(
        tmp_path,
        read={"type": f"{THIS}.Reads", "input": ["page"], "output": ["text"]},
    )
    assert {node["nodeType"] for node in nodes.values()} == {"Task", "Artefact"}


def test_an_artefact_node_lists_the_types_it_holds(tmp_path):
    nodes = graph(
        tmp_path,
        read={"type": f"{THIS}.Reads", "input": ["page"], "output": ["text"]},
        sort={"type": f"{THIS}.Sorts", "input": ["page"], "output": ["printed"]},
    )
    assert nodes["page"]["types"] == ["collectra.Image"]
    assert nodes["text"]["types"] == ["collectra.Text"]
    assert nodes["printed"]["types"] == ["collectra.Link"]
    assert nodes["printed"]["detail"] == "collectra.Link"


def test_a_task_node_lists_no_types(tmp_path):
    nodes = graph(
        tmp_path,
        read={"type": f"{THIS}.Reads", "input": ["page"], "output": ["text"]},
    )
    assert nodes["read"]["types"] == []
