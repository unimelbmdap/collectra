from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import yaml, uuid
from rich import print

from collectra.commons import BaseEntity, Node, NodeStatus
from collectra.utils import (
    error_msg,
    load_class_from_string,
    change_dir,
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

    def _generate_id(self) -> str:
        # Generate a unique ID based on the name and other attributes
        return f"{self.name}-{uuid.uuid4()}"

    def __post_init__(self) -> None:
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
                        assert data, f"{key} could not be found in {value}"
                        data = data if isinstance(data, list) else [data]
                        for item in data:           
                            primitive_type = False                                            
                            try:
                                if not isinstance(item, dict) or not (
                                    "type" in item and ("path" in item or "data" in item)
                                ):                      
                                    primitive_type = True                            
                                    raise Warning(
                                        f"Item does not have the correct data format: {item}"
                                    )
                                cls_ = load_class_from_string(item.pop("type"))
                                if not self.check_type(cls_):
                                    raise Warning(
                                        f"{cls_} is not a subclass or not defined in {self.types}"
                                    )
                                item["name"] = key
                                item["data"] = (
                                    item.pop("path") if "path" in item else item["data"]
                                )
                                if "parents" in item and not isinstance(item["parents"], list):
                                    item["parents"] = [item["parents"]]
                                if (
                                    validation is not None
                                    and "validation" in cls_.all_attributes()
                                ):
                                    item["validation"] = validation                                                                                             
                                instance = cls_(**item)
                                if not instance:
                                    raise Warning(f"Failed to load {item} with {cls_}")
                                self.add_item(instance)
                            except Exception as e:
                                traceback_error(e, f"Failed to load data item: {e}")                                
                                if primitive_type:
                                    print("Primitive type value found, loading it as Text...")
                                    for cls_ in self.types:
                                        try:
                                            instance = cls_(key, data=str(item))
                                            self.add_item(instance)                                            
                                        except Exception as e:
                                            traceback_error(e, f"Failed to load data: {e}")                                    

                except Exception as e:
                    traceback_error(e, f"Failed to load data: {e}")
        elif key:                     
            for cls_ in self.types:
                try:                                        
                    instance = cls_(key, data=value)
                    self.add_item(instance)
                except Exception as e:
                    traceback_error(e, f"Failed to load data: {e}")

    @staticmethod
    def batch_process(
        item_file: Path, data_nodes: list["DataNode"]
    ) -> list[Data]:        
        with change_dir(item_file):            
            try:
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
                all_names_not_found = all(name not in file_data for name in names)
                if all_names_not_found:
                    print(
                        f"[yellow]No matching data found in [blue]{item_file}[/blue] for names: {', '.join(names)}. Ignoring..."
                    )
                    found_base = False
                    for name, values in file_data.items():
                        values = values if isinstance(values, list) else [values]                        
                        for item in values:                                                                             
                            if not isinstance(item, dict) or "type" not in item and ("data" not in item or "path" not in item):
                                continue                                     
                            cls_str = item.pop("type", "")                            
                            if cls_str == "collectra.Image":                                
                                cls_ = load_class_from_string(cls_str)
                                try:
                                    item["name"] = name                        
                                    item["data"] = item.pop("path") if "path" in item else item["data"]
                                    if validation is not None:
                                        item["validation"] = validation
                                    instance = cls_(**item)                                    
                                    if not instance:
                                        raise Warning(f"Failed to load {item} with {cls_}")
                                    found_base = True
                                    data.append(instance)
                                    break
                                except Exception as e:
                                    traceback_error(
                                        e,
                                        f"Failed to load data item {name} from {item.get('data', '')}: {e}",
                                    )
                        if found_base:
                            break                    
                for i, name in enumerate(names):
                    if name not in file_data:                        
                        continue
                    value = file_data[name]
                    if not value:
                        raise Warning(f"Data seems to be empty for {name} in {item_file}. Provided: {value}")
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
                        item["data"] = item.pop("path") if "path" in item else item["data"]
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
                traceback_error(e, f"Failed to batch process data: {e}")            
                return list()