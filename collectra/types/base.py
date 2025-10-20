from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from collectra.commons import BaseEntity, Node, NodeStatus
from collectra.utils import error_msg, load_class_from_string, change_dir

__all__ = ["Data", "DataNode"]

@dataclass
class Data(BaseEntity):

    name: str    
    
    def serialize(self) -> dict:
        serialized = super().serialize()
        if "name" in serialized:
            serialized.pop("name")
        return serialized

@dataclass
class DataNode(Node):

    _items: list[Data] = field(default_factory=list)
    _types: set[type] = field(default_factory=set)

    @property
    def types(self) -> set[type]:
        return self._types

    @property
    def items(self) -> list[Data]:
        return self._items

    def __post_init__(self) -> None:
        self._status = NodeStatus.READY if self._items else NodeStatus.NOT_READY
    
    def add_item(self, item: Data) -> None:        
        self._items.append(item)
        self._status = NodeStatus.READY
    
    def add_type(self, type_: type) -> None:
        self._types.add(type_)
    
    def check_type(self, type_: type) -> bool:
        return any(issubclass(type_, t) for t in self._types)

    def __str__(self) -> str:        
        types_str = "\n".join([t.get_class_path() for t in self._types])
        return f"{self._name}\n{types_str}"

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
                        validation = results.pop("collectra_results_metadata", dict()).get("validation", None)       
                        data = results.get(key, None)
                        assert data, f"{key} could not be found in {value}"         
                        data = data if isinstance(data, list) else [data]                                                                       
                        for item in data:                                                        
                            if not isinstance(item, dict) or not ("type" in item and ("path" in item or "data" in item)):
                                raise ValueError(f"Item does not have the correct data format: {item}")                           
                            cls_ = load_class_from_string(item.pop("type"))                                        
                            if not self.check_type(cls_):
                                raise ValueError(f"{cls_} is not a subclass or not defined in {self.types}")
                            item["name"] = key
                            item["data"] = item.pop("path") if "path" in item else item["data"]
                            if validation is not None:
                                item["validation"] = validation                            
                            instance = cls_(**item)                                
                            if not instance:
                                raise ValueError(f"Failed to load {item} with {cls_}")
                            self.add_item(instance)                                                            
                except Exception as e:
                    print(error_msg(f"Failed to read data: {e}"))
        elif key:                   
            for cls_ in self.types:
                try:                    
                    instance = cls_(key, str(value))                    
                    self.add_item(instance)
                except Exception as e:
                    print(error_msg(f"Failed to load data item {key} from {value}: {e}"))
                    continue        

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
            validation = file_data.get("collectra_results_metadata", dict()).get("validation", None)        
        names = [data_node.name for data_node in data_nodes]        
        for i, name in enumerate(names):        
            if name not in file_data:
                continue
            value = file_data[name]
            value = value if isinstance(value, list) else [value]
            for item in value:                
                if not isinstance(item, dict) or not ("type" in item and "path" in item):
                    continue
                cls_ = load_class_from_string(item.pop("type"))            
                match = False
                for type_ in data_nodes[i]._types:
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
                    print(error_msg(f"Failed to load data item {name} from {item.get('data', '')}: {e}"))
                    continue            
        return data
        