from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from collectra.commons import BaseEntity, Node, NodeStatus
from collectra.utils import error_msg, load_class_from_string, change_dir, traceback_error

__all__ = ["Data", "DataNode"]


@dataclass
class Data(BaseEntity):

    name: str

    def attributes_to_ignore(self) -> set:
        attributes = super().attributes_to_ignore()
        attributes.add("name")
        return attributes

    @classmethod
    def all_attributes(cls):
        return set([attribute for attribute in dir(cls) if not attribute.startswith("_") and not callable(attribute)])


@dataclass
class DataNode(Node):

    items: list[Data] = field(default_factory=list)
    types: set[type] = field(default_factory=set)    

    def __post_init__(self) -> None:
        self.status = NodeStatus.READY if self.items else NodeStatus.NOT_READY

    def add_item(self, item: Data) -> None:
        self.items.append(item)
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

    def process(self, key: str, value: str | Path | None = None, **kwargs) -> None:
        value = value if value else kwargs.get("file", None)
        if value:
            value = Path(value)
        if value and value.exists() and value.is_dir():
            with change_dir(value):
                try:
                    result_file = Path("results.yaml")
                    with open(result_file, "r") as f:
                        results: dict = yaml.safe_load(f)
                        validation = results.pop(
                            "collectra_results_metadata", dict()
                        ).get("validation", None)
                        data = results.get(key, None)
                        assert data, f"{key} could not be found in {value}"
                        data = data if isinstance(data, list) else [data]
                        for item in data:
                            if not isinstance(item, dict) or not (
                                "type" in item and ("path" in item or "data" in item)
                            ):
                                raise ValueError(
                                    f"Item does not have the correct data format: {item}"
                                )
                            cls_ = load_class_from_string(item.pop("type"))
                            if not self.check_type(cls_):
                                raise ValueError(
                                    f"{cls_} is not a subclass or not defined in {self.types}"
                                )
                            item["name"] = key
                            item["data"] = (
                                item.pop("path") if "path" in item else item["data"]
                            )                            
                            if validation is not None and "validation" in cls_.all_attributes():
                                item["validation"] = validation                                                        
                            instance = cls_(**item)
                            if not instance:
                                raise ValueError(f"Failed to load {item} with {cls_}")
                            self.add_item(instance)
                except Exception as e:                    
                    traceback_error(e, f"Failed to load data: {e}")                    
        elif key:
            for cls_ in self.types:
                try:
                    instance = cls_(key, value)
                    self.add_item(instance)
                except Exception as e:
                    traceback_error(e, f"Failed to load data: {e}")                    
                    

    @staticmethod
    def batch_process(
        data_nodes: list["DataNode"],
    ) -> list[Data]:
        data: list[Data] = list()
        result_file = Path("results.yaml")
        if not result_file.exists():
            print(error_msg(f"Invalid file: {Path.cwd()}. Ignoring..."))
            return data
        with open(result_file, "r") as f:
            file_data: dict = yaml.safe_load(f)
            validation = file_data.get("collectra_results_metadata", dict()).get(
                "validation", None
            )
        names = [data_node.name for data_node in data_nodes]
        for i, name in enumerate(names):
            if name not in file_data:
                continue
            value = file_data[name]
            value = value if isinstance(value, list) else [value]
            for item in value:
                if not isinstance(item, dict) or not (
                    "type" in item and "path" in item
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
                item["data"] = item.pop("path", "")
                if validation is not None:
                    item["validation"] = validation
                try:
                    instance = cls_(**item)
                    if instance:
                        data.append(instance)
                except Exception as e:
                    traceback_error(e, f"Failed to load data item {name} from {item.get('data', '')}: {e}")                                        
        return data
