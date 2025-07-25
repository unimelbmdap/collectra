from __future__ import annotations
import uuid
from typing import List
from pathlib import Path
from rich import print
from rocrate.model.contextentity import ContextEntity
from ..engine.models import YOLOEngine, BaseEngine, EngineManager
from rocrate.rocrate import ROCrate

# TODO: Move this to a task library module ----

class TaskRepo:      
  
  @staticmethod
  def detect_objects(config: dict = {}):
    engine = YOLOEngine(
      
    )
    print(f"[bold green]Running object detection with engine[/bold green]: {engine.model}")
    engine.run()      

class TaskDefinition:
  def __init__(self, task_type: str, description: str):
    self.task_type = task_type
    self.description = description
  
  def __str__(self):
    return f"{self.task_type}: {self.description}"

TASK_DEFINITIONS: dict = {
  "convert_annotation": TaskDefinition(
    task_type="convert_annotation",
    description="Convert annotations from one format to another",
  ),
  "detect_objects": TaskDefinition(
    task_type="detect_objects",
    description="Detect objects in images using a specified engine",
  )
}

#---------------------------------------------

class Task(ContextEntity):
  def __init__(self, crate, identifier=None, properties=None):
    super(Task, self).__init__(crate, identifier, properties)
  
  def _empty(self):
    return {
        "@id": self.id,
        "@type": "Task",
        "description": "",                
        "task_type": "",        
    } 

class TaskModel:
  def __init__(self, task_definition: TaskDefinition, config: dict = {}):    
    self.id = uuid.uuid4()    
    self.task_definition = task_definition    
    self.config = config    
    self.engine = None
  
  def to_crate(self, crate: ROCrate):
    """
    Convert the TaskModel to a ROCrate entity.
    :param crate: The ROCrate instance to add the task to.
    :return: The created Task entity.
    """
    if not self.engine:
      raise ValueError("Engine must be set before converting to crate.")
    task_crate = Task(crate, identifier=self.id, properties={
        "description": self.describe(),
        "task_type": self.task_definition.task_type,                
    })    
    task_crate["engine"] = [self.engine.to_crate(crate)]
    crate.add(task_crate)
    return task_crate

  def add_engine(self, engine: BaseEngine):
    """
    Add an engine to the task.
    :param engine: The EngineModel instance to add.
    """    
    self.engine = engine
  
  def run(self):
    if not self.engine:
      raise ValueError("Engine must be set before running the task.")
    print(f"[bold green]Running task[/bold green]: {self.task_definition.task_type}")
    print(f"Using engine: {self.engine.name} {self.engine.version}")
    self.engine.run(self.config)

  def __str__(self):
    return f"{self.task_definition.task_type}"

  def describe(self):
    description = self.task_definition.description if self.task_definition.description else "No description available."
    return description       

class TaskManager:

  def __init__(self, task_chain: List[TaskModel] = []):
    self.task_chain = task_chain    

  def run(self):
    for task in self.task_chain:
      task.run()
  
  def chain(self, task: Task, config: dict = {}) -> TaskModel:
    if not task:
      return None         
    task_type = task["task_type"]
    engine = task["engine"][0]
    engine_model = EngineManager.get(engine["name"])
    # engine_model = BaseEngine(
    #   name=engine["name"],
    #   version=engine["version"],
    #   source=engine.get("source", "default"),
    #   description=engine.get("description", ""),
    #   config=engine.get("config", {})
    # )    
    task = TaskModel(TASK_DEFINITIONS[task_type], config=config)
    task.add_engine(engine_model)
    self.task_chain.append(task)
    return task    

  @staticmethod
  def build(task: str, crate: ROCrate = None) -> TaskModel:
    """
    Build a Task from a string string.
    :param task: The task string in the format "<task_type>,<engine>".
    :param crate: The ROCrate instance to add the task to.
    :raises ValueError: If the task format is invalid or the task type is not recognized
    :return: An instance of EngineModel.
    """
    if "," not in task:
      raise ValueError(f"[bold red]Invalid task format[/bold red]: {task}. Must be in the format <task_type>,<engine>.")
    task_type, engine = task.split(",")
    if task_type not in TASK_DEFINITIONS:
      raise ValueError(f"[bold red]Invalid task type[/bold red]: {task_type}. Must be one of {list(TASK_DEFINITIONS.keys())}.")
    engine = EngineManager.build(engine)
    task: TaskModel = TaskModel(TASK_DEFINITIONS[task_type])
    task.add_engine(engine)
    print(engine)
    task.to_crate(crate)
    return task
