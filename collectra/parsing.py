from pathlib import Path
import yaml
import networkx as nx
import importlib
from dataclasses import dataclass, field

@dataclass(kw_only=True)
class CollectraItem():
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
class CollectraTask(CollectraItem):
    def set_node_attributes(self, node):
        node["color"] = "dodgerblue"
        node["shape"] = "box"


@dataclass(kw_only=True)
class TextProcessor(CollectraTask):
    pass


def load_class_from_string(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def read_collectra(path:Path|str) -> nx.DiGraph:
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f)

    G = nx.DiGraph()

    items = dict()

    for name, kwargs in data.items():
        type_ = kwargs.pop("type", None)
        assert type_ is not None, f"Type is required for {name}"

        cls = load_class_from_string(type_)
        item = cls(name=name, **kwargs)
        assert isinstance(item, CollectraItem)
        items[name] = item

        if name not in G:
            G.add_node(name, item=item)
                    
        node = G.nodes[name]
        node["item"] = item

        # Set colour and attributes of node in networkx
        item.set_node_attributes(node)

        for input_name in item.input:
            G.add_edge(input_name, name)

        for output_name in item.output:
            G.add_edge(name, output_name)

    return G


def render_collectra(path:Path|str, output:Path|str|None=None) -> str:
    G = read_collectra(path)
    dot_string = nx.nx_pydot.to_pydot(G).to_string()
    if output:
        import graphviz

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        graph = graphviz.Source(dot_string)
        format = output.suffix[1:] if output.suffix else "svg"
        graph.render(str(output.with_suffix('')), format=format, cleanup=True)

    return dot_string