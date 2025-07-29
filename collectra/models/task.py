from __future__ import annotations
from abc import ABC, abstractmethod
import uuid
from typing import Tuple
from rich import print
from rocrate.model.data_entity import DataEntity
from .engine import *
from rocrate.rocrate import ROCrate


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

  VALID_ENGINES: Tuple = ()  

  def __init__(self, task_type: str, config: dict = {}, engine: Engine = None, id: str = None):
    self.id = uuid.uuid4() if id is None else id
    self.task_type: str = task_type         
    self.config = config    
    self.engine = None    
    if self.validate_engine(engine):      
      self.engine = engine
  
  def __str__(self) -> str:
    return f"{self.task_type}"       

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
  
  def validate_engine(self, engine: Engine) -> bool:        
        valid_engine: bool = isinstance(engine, self.VALID_ENGINES)
        if not valid_engine:
            print(
              f"Invalid engine type: {type(engine).__name__}. "
              f"Valid engines are: {[e.__name__ for e in self.VALID_ENGINES]}."
            )        
        return valid_engine
  
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
  def validate(self) -> None:
    pass
  

class DetectObject(Task):

  VALID_ENGINES: Tuple = (
      YOLOEngine,
      DETECTRON2Engine,
  )  

  def run(self) -> None:
    if not self.engine:
      raise ValueError("Engine must be set before running the task.")
    self.engine.run(self.config)
  
  def train(self) -> None:
    if not self.engine:
      raise ValueError("Engine must be set before training the task.")
    return self.engine.train(self.config)
  
  def validate(self) -> None:
    if not self.engine:
      raise ValueError("Engine must be set before validating the task.")
    self.engine.validate(self.config)


class ClassifyImage(Task):

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


  def validate(self) -> None:
    if not self.engine:
      raise ValueError("Engine must be set before validating the task.")
    self.engine.validate(self.config)
        
  



  

  

