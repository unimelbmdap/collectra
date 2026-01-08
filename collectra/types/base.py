import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from rich import print

from ..commons import BaseEntity, Node, NodeStatus
from ..utils import (
    change_dir,
    error_msg,
    load_class_from_string,
    traceback_error,
)

__all__ = ["Data", "DataNode"]


@dataclass
class Data(BaseEntity):

    name: str
    id: str = field(default="")
    parents: list[str] = field(default_factory=list)
    validation: bool = field(default=False)

    def set_parents(self, parents: list["Data"]) -> None:
        self.parents = [parent.id for parent in parents]

    def serialize(self) -> dict:
        serialized = super().serialize()
        if "parents" in serialized:
            if len(serialized["parents"]) == 0:
                serialized.pop("parents")
            elif len(serialized["parents"]) == 1:
                serialized["parents"] = serialized["parents"][0]
        return serialized

    def eval(self, gold: "Data") -> dict:
        raise NotImplementedError("Eval method not implemented for base Data class.")

    def _generate_id(self) -> str:
        # Generate a unique ID based on the name and other attributes
        return f"{self.name}-{uuid.uuid4()}"

    def __post_init__(self) -> None:
        super().__init__(self.name)  # Initialize BaseEntity
        if not self.id:
            self.id = self._generate_id()

    def attributes_to_ignore(self) -> set:
        attributes = super().attributes_to_ignore()
        attributes.add("name")
        attributes.add("validation")
        return attributes

    @classmethod
    def all_attributes(cls):
        return set(
            [
                attribute
                for attribute in dir(cls)
                if not attribute.startswith("_") and not callable(attribute)
            ]
        )


@dataclass
class DataNode(Node):

    items: dict[str, Data] = field(default_factory=dict)
    types: set[type] = field(default_factory=set)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.status = NodeStatus.READY if self.items else NodeStatus.NOT_READY

    def add_item(self, item: Data) -> None:
        self.items[item.id] = item
        self.status = NodeStatus.READY

    def add_type(self, type_: type) -> None:
        self.types.add(type_)

    def check_type(self, type_: type) -> bool:
        for t in self.types:
            if issubclass(type_, t):
                return True
        return False

    def __str__(self) -> str:
        types_str = "\n".join([t.get_class_path() for t in self.types])
        return f"{self.name}\n{types_str}"

    def eval(self, gold_items: "DataNode"):
        evaluation_matrix = np.array(
            [[0.0 for _ in gold_items.items.items()] for _ in self.items.items()]
        )
        for item_key, item in self.items.items():
            for gold_key, gold_item in gold_items.items.items():
                evaluation_matrix[self.items[item_key], gold_items.items[gold_key]] = (
                    item.eval(gold_item)
                )
        # Start finding the highest scores and matching them
        matched_items = []
        while len(matched_items) < len(gold_items.items.items()):
            max_val = evaluation_matrix.max()
            item_idx, gold_idx = np.unravel_index(
                evaluation_matrix.argmax(), evaluation_matrix.shape
            )

    def _create_instance(self, cls_: type, **item) -> None:
        try:
            instance = cls_(**item)
            if not instance:
                raise ValueError(f"Failed to load {item} with {cls_}")
            self.add_item(instance)
        except Exception as e:
            self.catcher.set_err(str(e))

    def _create_instances(self, **item) -> None:
        for cls_ in self.types:
            self._create_instance(cls_, **item)

    def process(self, key: str, value: str | Path | None = None, **kwargs) -> None:
        value = value if value else kwargs.get("file", None)
        if value and Path(value).exists() and Path(value).is_dir():
            value = Path(value)
            with change_dir(value):
                try:
                    result_file = Path("results.yaml")
                    with open(result_file, "r") as f:
                        results: dict = yaml.safe_load(f)
                        validation = results.pop(
                            "collectra_results_metadata", dict()
                        ).get("validation", None)
                        data = results.get(key, None)
                        if not data:
                            raise ValueError(f"[red]{key}[/red] could not be found in {value}")
                        data = data if isinstance(data, list) else [data]
                        for item in data:
                            primitive_type = False
                            try:
                                primitive_type = not isinstance(item, dict) or not (
                                    "type" in item
                                    and ("path" in item or "data" in item)
                                )
                                if primitive_type:
                                    raise ValueError(f"Item must be a complex type with 'type' and ('path' or 'data')")
                                cls_ = load_class_from_string(item.pop("type"))
                                if not self.check_type(cls_):
                                    raise TypeError(f"{cls_} is not a subclass or not defined in {self.types}")
                                item["name"] = key
                                if "data" not in item:
                                    item["data"] = item.pop("path")
                                if "parents" in item and not isinstance(
                                    item["parents"], list
                                ):
                                    item["parents"] = [item["parents"]]
                                if (
                                    validation is not None
                                    and "validation" in cls_.all_attributes()
                                ):
                                    item["validation"] = validation
                                self._create_instance(cls_, **item)
                            except Exception as e:
                                if not primitive_type:
                                    self.catcher.set_err(str(e))
                                else:
                                    self._create_instances(name=key, data=str(item))

                except Exception as e:
                    self.catcher.set_err(str(e))
        elif key:
            self._create_instances(name=key, data=value)

    @staticmethod
    def batch_process(item_file: Path, data_nodes: list["DataNode"]) -> list[Data]:
        with change_dir(item_file):
            try:
                data: list[Data] = list()
                result_file = Path("results.yaml")
                if not result_file.exists():
                    print(error_msg(f"Invalid file: {Path.cwd()}. Ignoring..."))
                    return data
                with open(result_file, "r") as f:
                    file_data: dict = yaml.safe_load(f)
                    validation = file_data.get(
                        "collectra_results_metadata", dict()
                    ).get("validation", None)
                names = [data_node.name for data_node in data_nodes]
                all_names_not_found = all(name not in file_data for name in names)
                if all_names_not_found:
                    print(
                        f"[yellow]No matching data found in [blue]{item_file}[/blue] for names: {', '.join(names)}. Ignoring..."
                    )
                    found_base = False
                    for name, values in file_data.items():
                        values = values if isinstance(values, list) else [values]
                        for item in values:
                            if (
                                not isinstance(item, dict)
                                or "type" not in item
                                and ("data" not in item or "path" not in item)
                            ):
                                continue
                            cls_str = item.pop("type", "")
                            if cls_str == "collectra.Image":
                                cls_ = load_class_from_string(cls_str)
                                try:
                                    item["name"] = name
                                    item["data"] = (
                                        item.pop("path")
                                        if "path" in item
                                        else item["data"]
                                    )
                                    if validation is not None:
                                        item["validation"] = validation
                                    instance = cls_(**item)
                                    if not instance:
                                        raise Warning(
                                            f"Failed to load {item} with {cls_}"
                                        )
                                    found_base = True
                                    data.append(instance)
                                    break
                                except Exception as e:
                                    traceback_error(
                                        e,
                                        f"Failed to load data item {name} from {item.get('data', '')}",
                                        verbose=True,
                                    )
                        if found_base:
                            break
                for i, name in enumerate(names):
                    if name not in file_data:
                        continue
                    value = file_data[name]
                    if not value:
                        raise Warning(
                            f"Data seems to be empty for {name} in {item_file}. Provided: {value}"
                        )
                    value = value if isinstance(value, list) else [value]
                    for item in value:
                        if not isinstance(item, dict) or not (
                            "type" in item and ("data" in item or "path" in item)
                        ):
                            continue
                        cls_ = load_class_from_string(item.pop("type"))
                        match = False
                        for type_ in data_nodes[i].types:
                            if issubclass(cls_, type_) or cls_ == type_:
                                match = True
                                break
                        if not match:
                            continue
                        item["name"] = name
                        item["data"] = (
                            item.pop("path") if "path" in item else item["data"]
                        )
                        if validation is not None:
                            item["validation"] = validation
                        try:
                            instance = cls_(**item)
                            if instance:
                                data.append(instance)
                        except Exception as e:
                            traceback_error(
                                e,
                                f"Failed to load data item {name} from {item.get('data', '')}: {e}",
                            )
                return data
            except Exception as e:
                traceback_error(e, verbose=True)
                return list()
