from __future__ import annotations
import uuid
from typing import List
from pathlib import Path
from rich import print
from rocrate.model.contextentity import ContextEntity
from .engine import *
from rocrate.rocrate import ROCrate
from enum import Enum
from abc import ABC, abstractmethod

#---------------------------------------------

class TaskEntity(ContextEntity):
  def __init__(self, crate, identifier=None, properties=None):
    super(TaskEntity, self).__init__(crate, identifier, properties)
  
  def _empty(self):
    return {
        "@id": self.id,
        "@type": "Task",
        "description": "",                
        "task_type": "",        
    } 
  

class Task(ABC):

  def __init__(self, task_type: str, config: dict = {}, engine: Engine = None):
    self.id = uuid.uuid4()   
    self.task_type: str = task_type         
    self.config = config    
    self.engine = None
    if self.validate_engine(engine):
      self.engine = engine
  
  def __str__(self) -> str:
    return f"{self.task_definition.task_type}"       
  
  def to_crate(self, crate: ROCrate) -> ROCrate:
    """
    Convert the TaskModel to a ROCrate entity.
    :param crate: The ROCrate instance to add the task to.
    :return: The created Task entity.
    """
    if not self.engine:
      raise ValueError("Engine must be set before converting to crate.")
    task_crate = TaskEntity(crate, identifier=self.id, properties={
        "description": self.describe(),
        "task_type": self.task_definition.task_type,                
    })    
    task_crate["engine"] = [self.engine.to_crate(crate)]
    crate.add(task_crate)
    return task_crate
  
  def validate_engine(self, engine: Engine) -> bool:
        valid_engine = isinstance(engine, self.VALID_ENGINES)
        if not valid_engine:
            print(
              f"Invalid engine type: {type(engine).__name__}. "
              f"Valid engines are: {[e.__name__ for e in self.VALID_ENGINES]}."
            )
        return valid_engine
            
  def add_engine(self, engine: Engine) -> None:
    """
    Add an engine to the task.
    :param engine: The EngineModel instance to add.
    """ 
    self.validate_engine(engine)   
    self.engine = engine
  
  @abstractmethod
  def run(self) -> None:
    pass

  @abstractmethod
  def train(self) -> None:
    pass

  @abstractmethod
  def validate(self) -> None:
    pass

  @abstractmethod
  def describe(self) -> str:
    pass

class DetectObject(Task):

    VALID_ENGINES = (
        YOLOEngine,
        DETECTRON2Engine,
    )                 

class ClassifyImage(Task):

    VALID_ENGINES = (
        ImageClassifier,
    )
        



  

  

