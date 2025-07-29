from .task import Task, TaskEntity, DetectObject, ClassifyImage
from .engine import Engine, EngineEntity, YOLOEngine, ImageClassifier, DETECTRON2Engine
from rich import print
from typing import List
from rocrate.rocrate import ROCrate
from pathlib import Path
import os

class EngineManager:    

  ENGINE_TYPES: dict = {
    "yolo": YOLOEngine,
    "image_classifier": ImageClassifier,
    "detectron2": DETECTRON2Engine,  # Placeholder for future implementation
  }

  @staticmethod
  def build(type: str, name: str) -> Engine:
    EngineClass = EngineManager.ENGINE_TYPES.get(type)
    if not EngineClass:
      raise ValueError(f"[bold red]Invalid engine type[/bold red]: {type}. Must be one of {list(EngineManager.ENGINE_TYPES.keys())}.")
    return EngineClass(name)
  
  @staticmethod
  def get(engine: EngineEntity) -> Engine:    
    EngineClass = EngineManager.ENGINE_TYPES.get(engine.get("engine_type"))    
    if not EngineClass:
      raise ValueError(f"[bold red]Invalid engine type[/bold red]: {engine.get("engine_type")}. Must be one of {list(EngineManager.ENGINE_TYPES.keys())}.")
    engine.write(Path(f"tmp"))
    return EngineClass, Path(f"tmp/{engine.id}")

class TaskManager:  

  VALID_TASKS: dict = {
    "detect_object": DetectObject,
    "classify_image": ClassifyImage,
  }  

  def __init__(self, task_chain: List[Task] = []):
    self.task_chain = task_chain    

  def run(self):
    for task in self.task_chain:
      task.run()

  def list_tasks() -> List[str]:
    """
    List all available tasks.
    :return: A list of task names.
    """
    return list(TaskManager.VALID_TASKS.keys())
  
  def chain(self, task: TaskEntity, config: dict = {}) -> Task:
    if not task:
      return None         
    task_type = task["task_type"]
    engine = task["engine"][0]
    engine_model = EngineManager.build(engine["name"])
    task = Task(TaskManager.TASK_DEFINITIONS[task_type], config=config)
    task.add_engine(engine_model)
    self.task_chain.append(task)
    return task    
  
  @staticmethod
  def build(task: str, crate: ROCrate = None) -> Task:
    """
    Build a Task from a string string.
    :param task: The task string in the format "<task_type>,<engine>".
    :param crate: The ROCrate instance to add the task to.
    :raises ValueError: If the task format is invalid or the task type is not recognized
    :return: An instance of EngineModel.
    """
    if "," not in task:
      raise ValueError(f"[bold red]Invalid task format[/bold red]: {task}. Must be in the format <task_type>,<engine>.")
    task_type, engine_type, engine = task.split(",")
    ValidTask = TaskManager.VALID_TASKS.get(task_type)
    if not ValidTask:
      raise ValueError(f"[bold red]Invalid task type[/bold red]: {task_type}. Must be one of {list(TaskManager.VALID_TASKS.keys())}.")
    ValidEngine = EngineManager.build(engine_type, engine)
    task = DetectObject(task_type=task_type, engine=ValidEngine)                
    task.to_crate(crate)
    return task
  
  @staticmethod
  def get(task: str, crate: ROCrate = None) -> Task:
    """
    Get a Task from a string string.
    :param task: The task string in the format "<task_type>,<engine>".
    :param crate: The ROCrate instance to add the task to.
    :raises ValueError: If the task format is invalid or the task type is not recognized
    :return: An instance of Task.
    """
    if "," not in task:
      raise ValueError(f"[bold red]Invalid task format[/bold red]: {task}. Must be in the format <task_type>,<engine>.")
    return TaskManager.build(task, crate)

  @staticmethod
  def train(task: str, config: dict = {}) -> Task:    
    """
    Get a Task from a string string.
    :param task: The task string in the format "<task_type>,<engine>".
    :param crate: The ROCrate instance to add the task to.
    :raises ValueError: If the task format is invalid or the task type is not recognized
    :return: An instance of Task.
    """
    crate = config.get("crate")
    os.makedirs(config.get("tmp_dir", "tmp"), exist_ok=True)    
    if crate is None:
      raise ValueError("[bold red]Crate must be provided[/bold red] when getting a task.")
    file_format = crate.dereference("./").get("file_format", "grapto").replace(".", "")
    for e in crate.data_entities:
      if e.type == "Task":
        ValidTask = TaskManager.VALID_TASKS.get(e.get("task_type"))
        if not ValidTask:
          print(f"[bold red]Invalid task type[/bold red]: {e.get('task_type')}")
          continue        
        engine = e.get("engine")[0]
        training_params = dict(engine.get("trainingParameters")[0]) | {}
        config = {
          **config,
          **training_params,
          "file_format": file_format,
        }
        ValidEngine, engine_path = EngineManager.get(engine)                
        print(f"[bold green]Found task[/bold green]: {ValidTask}")
        print(f"[bold green]Found engine[/bold green]: {ValidEngine}")            
        print(f"[bold green]Using config[/bold green]: {config}")        
        print(f"[bold green]Using engine path[/bold green]: {engine_path}")
        engine = ValidEngine(name=engine_path)
        task = ValidTask(task_type=e.get("task_type"), config=config, engine=engine, id=e.id)        
        new_engine_path = task.train()       
        engine = ValidEngine(name=new_engine_path)        
        task.add_engine(engine, crate)
        task_crate = task.to_crate(crate)        
        print("[green]New engine added:[/green]", task_crate.get("engine")[0].id)        
  