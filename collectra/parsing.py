from pathlib import Path
import yaml
import networkx as nx
import importlib
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class CollectraNode:
    name: str
    input: list[str] = field(default_factory=list)
    output: list[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.input, str):
            self.input = [self.input]
        if isinstance(self.output, str):
            self.output = [self.output]

    def set_node_attributes(self, node):
        pass


@dataclass(kw_only=True)
class CollectraTask(CollectraNode):
    def set_node_attributes(self, node):
        node["color"] = "dodgerblue"
        node["shape"] = "box"


@dataclass(kw_only=True)
class TextProcessor(CollectraTask):
    pass


@dataclass(kw_only=True)
class ObjectDetection(CollectraTask):
    pass


@dataclass(kw_only=True)
class OCR(CollectraTask):
    pass


def load_class_from_string(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


@dataclass()
class CollectraWorkflow:
    path: Path
    dag: nx.DiGraph = field(init=False, default=None)

    def __post_init__(self):
        self.read_yaml()

    def read_yaml(self) -> nx.DiGraph:
        path = Path(self.path)
        with path.open("r") as f:
            data = yaml.safe_load(f)

        self.dag = nx.DiGraph()

        items = dict()

        for name, kwargs in data.items():
            type_ = kwargs.pop("type", None)
            assert type_ is not None, f"Type is required for {name}"

            cls = load_class_from_string(type_)
            item = cls(name=name, **kwargs)
            assert isinstance(item, CollectraNode)
            items[name] = item

            if name not in self.dag:
                self.dag.add_node(name, item=item)

            node = self.dag.nodes[name]
            node["item"] = item

            # Set colour and attributes of node in networkx
            item.set_node_attributes(node)

            for input_name in item.input:
                self.dag.add_edge(input_name, name)

            for output_name in item.output:
                self.dag.add_edge(name, output_name)

        return self.dag

    def dot(self) -> str:
        return nx.nx_pydot.to_pydot(self.dag).to_string()

    def render(self, output: Path | str) -> str:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        dot_string = self.dot()
        suffix = output.suffix.lower()
        if suffix == ".dot":
            with output.open("w") as f:
                f.write(dot_string)
        else:
            import graphviz

            graph = graphviz.Source(dot_string)
            format = suffix[1:] if suffix else "svg"
            graph.render(str(output.with_suffix("")), format=format, cleanup=True)

        return dot_string
