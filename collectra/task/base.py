from __future__ import annotations
import uuid
from typing import List
from pathlib import Path
from rich import print
from rocrate.model.contextentity import ContextEntity
from .engine import YoloEngine, Engine, EngineManager
from rocrate.rocrate import ROCrate
from enum import Enum
from abc import ABC, abstractmethod

class TaskRepo:      
  @staticmethod
  def detect_objects(config: dict = {}):
    engine = YoloEngine()
    print(f"[bold green]Running object detection with engine[/bold green]: {engine.model}")
    engine.run()      

class TaskDefinition:
  def __init__(self, task_type: str, description: str):
    self.task_type = task_type
    self.description = description
  
  def __str__(self):
    return f"{self.task_type}: {self.description}"

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

  def __init__(self, task_definition: TaskDefinition, config: dict = {}):    
    self.id = uuid.uuid4()    
    self.task_definition = task_definition    
    self.config = config    
    self.engine = None

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

  def add_engine(self, engine: Engine) -> None:
    """
    Add an engine to the task.
    :param engine: The EngineModel instance to add.
    """    
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
  def check_valid_engine(self, engine: Engine) -> None:
    pass

  @abstractmethod
  def describe(self) -> str:
    pass

  

  

  

