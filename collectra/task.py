from __future__ import annotations
from abc import ABC, abstractmethod
import uuid
from typing import Tuple
from rich import print
from rocrate.model.data_entity import DataEntity
from rocrate.rocrate import ROCrate
from .engine import *

#---------------------------------------------

class TaskEntity(DataEntity):
  def __init__(self, crate, identifier=None, properties=None):
    super(TaskEntity, self).__init__(crate, identifier, properties)
  
  def _empty(self):
    return {
        "@id": self.id,
        "@type": "Task",                     
        "task_type": "",        
    } 
  
class Task(ABC):

  VALID_ENGINES: dict = {}
  VALID_INPUTS = []
  VALID_OUTPUTS = []

  def __init__(self, id: str = None, engine_path: str = None, engine_type: str = None, inputs: list = [], outputs: list = [], config: dict = {}):
    self.id = id if id else uuid.uuid4()            
    self.engine = self.validate_engine(engine_path, engine_type)            
    self.inputs = inputs if len(inputs) > 0 else self.VALID_INPUTS
    self.outputs = outputs if len(outputs) > 0 else self.VALID_OUTPUTS        
    self.config = config

  def get_metadata(self) -> dict:
    """
    Get metadata of the task.
    :return: A dictionary containing task metadata.
    """
    return {
        "id": self.id,        
        "engine": self.engine.name if self.engine else None,
        "config": self.config,
    }
  
  def __str__(self) -> str:
    return f"{self.id} of type {self.__class__.__name__}"       

  def to_crate(self, crate: ROCrate) -> ROCrate:
    """
    Convert the TaskModel to a ROCrate entity.
    :param crate: The ROCrate instance to add the task to.
    :return: The created Task entity.
    """    
    task_crate: TaskEntity = TaskEntity(crate, identifier=self.id, properties={        
        "task_type": self.task_type,                
    })     
    if self.engine:   
      task_crate["engine"] = [self.engine.to_crate(crate)]
    crate.add(task_crate)
    return task_crate

  def validate_engine(self, engine_path: str, engine_type: str) -> bool:    
    if engine_type not in self.VALID_ENGINES:
        print(f"[bold red]Invalid engine type[/bold red]: {engine_type}. Must be one of {list(self.VALID_ENGINES.keys())}.")
        return None
    engine = self.VALID_ENGINES.get(engine_type)(engine_path)            
    return engine
  
  def slug(self) -> str:
    return self.task_type
            
  def add_engine(self, engine: Engine, crate: ROCrate = None) -> bool:
    """
    Add an engine to the task.
    :param engine: The EngineModel instance to add.
    """ 
    validated = self.validate_engine(engine)    
    if crate and self.engine:
      old_engine_id = f"{self.engine.name.stem.replace('tmp/', '')}{self.engine.name.suffix}"
      old_engine = crate.dereference(old_engine_id)            
      crate.delete(old_engine)
      print(f"[bold red]Deleted old engine[/bold red]: {old_engine_id}")
    if validated:
        self.engine: Engine = engine
    return validated

  @abstractmethod
  def run(self) -> None:
    pass

  @abstractmethod
  def train(self) -> None:
    pass

  @abstractmethod
  def eval(self) -> None:
    pass
  

class ObjectDetection(Task):

  VALID_ENGINES: dict = {
      "yolo": YOLOEngine,
      "detectron2": DETECTRON2Engine,
  }      

  VALID_INPUTS = ["specimen_sheet"]
  VALID_OUTPUTS = ["primary_specimen_label", "handwritten_data",
                   "annotation_label", "stamp", "swing_tag",
                   "accession_number", "small_database_label",
                   "medium_database_label", "full_database_label",
                   "swatch", "scale"]

  def run(self) -> None:
    if not self.engine:
      raise ValueError("Engine must be set before running the task.")
    self.engine.run(self.config)
  
  def train(self) -> None:
    if not self.engine:
      raise ValueError("Engine must be set before training the task.")
    return self.engine.train(self.config)
  
  def eval(self) -> None:
    if not self.engine:
      raise ValueError("Engine must be set before validating the task.")
    self.engine.val(self.config)
  
  def cluster(self) -> None:
    if not self.engine:
      raise ValueError("Engine must be set before clustering the task.")
    self.engine.cluster(self.config)

class TextClassification(Task):

  VALID_ENGINES: Tuple = (
      ImageClassifier,
  )

  def __init__(self, task_type: str = "classify_image", config: dict = {}, engine: Engine = None):
    super().__init__(task_type, config, engine)
  
  def run(self) -> None:
    if not self.engine:
      raise ValueError("Engine must be set before running the task.")
    self.engine.run(self.config)
  
  def train(self) -> Path:
    if not self.engine:
      raise ValueError("Engine must be set before training the task.")
    return self.engine.train(self.config)


  def eval(self) -> None:
    if not self.engine:
      raise ValueError("Engine must be set before validating the task.")
    self.engine.val(self.config)
        
  



  

  

