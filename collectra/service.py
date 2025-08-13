from .task import Task, TaskEntity, ObjectDetectionYOLO, TextClassification
from .model import Model, EngineEntity, YOLOModel, ImageClassifier, DETECTRON2Engine
import typer
from rich import print
from typing import List, Union
from rocrate.rocrate import ROCrate
from pathlib import Path
import os

class EngineManager:    

  ENGINE_TYPES: dict = {
    "yolo": YOLOModel,
    "image_classifier": ImageClassifier,
    "detectron2": DETECTRON2Engine,  # Placeholder for future implementation
  }

  @staticmethod
  def build(type: str, name: str) -> Model:
    EngineClass = EngineManager.ENGINE_TYPES.get(type)
    if not EngineClass:
      raise ValueError(f"[bold red]Invalid engine type[/bold red]: {type}. Must be one of {list(EngineManager.ENGINE_TYPES.keys())}.")
    return EngineClass(name)
  
  @staticmethod
  def get(engine: EngineEntity) -> Model:    
    EngineClass = EngineManager.ENGINE_TYPES.get(engine.get("engine_type"))    
    if not EngineClass:
      raise ValueError(f"[bold red]Invalid engine type[/bold red]: {engine.get('engine_type')}. Must be one of {list(EngineManager.ENGINE_TYPES.keys())}.")
    engine.write(Path(f"tmp"))
    return EngineClass, Path(f"tmp/{engine.id}")

class TaskManager:  

  VALID_TASKS: dict = {
    "detect_object": ObjectDetectionYOLO,
    "classify_image": TextClassification,
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
    typer.echo(f"building task string: {task}")
    if "," not in task:
      raise ValueError(f"[bold red]Invalid task format[/bold red]: {task}. Must be in the format <task_type>,<task_name>,<engine_type>,<engine>.")
    task_type,task_name,engine_type,engine = task.split(",")
    ValidTask = TaskManager.VALID_TASKS.get(task_type)
    if not ValidTask:
      raise ValueError(f"[bold red]Invalid task type[/bold red]: {task_type}. Must be one of {list(TaskManager.VALID_TASKS.keys())}.")
    ValidEngine = EngineManager.build(engine_type, engine)
    task = ObjectDetectionYOLO(task_type=task_type, engine=ValidEngine, id=task_name)                
    task.to_crate(crate)
    return task
  
  @staticmethod
  def edit(task: str, crate: ROCrate, param: str, value: Union[str, int, float]) -> Task:
    """
    Edit a Task's parameter.
    """
    task = crate.dereference(task)
    print(task)
    if task.type != "Task":
      raise ValueError(f"[bold red]Task not found[/bold red]: {task}. Must be a valid TaskEntity.")
    engine = task.get("engine")[0]
    if not engine or not engine.type == "File":
      raise ValueError(f"[bold red]Engine not found[/bold red] for task: {task.id}. Must be a valid EngineEntity.")
    training_params = engine.get("trainingParameters")[0]
    if not training_params or not training_params.type == "TrainingParameters":
      raise ValueError(f"[bold red]No training parameters found[/bold red] for task: {task.id}.")
    if param not in training_params:
      raise ValueError(f"[bold red]Parameter not found[/bold red]: {param}. Must be one of {list(training_params.keys())}.")
    training_params[param] = int(value) if isinstance(value, str) and value.isdigit() else value
    return task
  
  @staticmethod
  def get(task: str, config: dict) -> Task:
    """
    Get a Task from a string string.
    :param task: The task string in the format "<task_type>,<engine>".
    :param crate: The ROCrate instance to add the task to.
    :raises ValueError: If the task format is invalid or the task type is not recognized
    :return: An instance of Task.
    """
    crate: ROCrate = config.get("crate")
    task: TaskEntity = crate.dereference(task)
    if task.type != "Task":
      raise ValueError(f"[bold red]Task not found[/bold red]: {task}. Must be a valid TaskEntity.")    
    task_type: str = task.get("task_type")
    ValidTask = TaskManager.VALID_TASKS.get(task_type)
    if not ValidTask:
      raise ValueError(f"[bold red]Invalid task type[/bold red]: {task.get('task_type')}. Must be one of {list(TaskManager.VALID_TASKS.keys())}.")    
    engine: EngineEntity = task.get("engine")[0]
    training_params = dict(engine.get("trainingParameters")[0]) | {}
    config = {
      **config,
      **training_params,      
    }
    ValidEngine, engine_path = EngineManager.get(engine)
    engine = ValidEngine(name=engine_path)    
    task = ValidTask(task_type=task_type, config=config, engine=engine, id=task.id)
    print(f"[bold green]Found task[/bold green]: {Task}")
    print(f"[bold green]Found engine[/bold green]: {ValidEngine}")    
    print(f"[bold green]Using engine path[/bold green]: {engine_path}")
    return task, ValidEngine

  @staticmethod
  def train(task: str, config: dict = {}) -> Task:    
    """
    Get a Task from a string string.
    :param task: The task string in the format "<task_type>,<engine>".
    :param crate: The ROCrate instance to add the task to.
    :raises ValueError: If the task format is invalid or the task type is not recognized
    :return: An instance of Task.
    """
    crate: ROCrate = config.get("crate")
    os.makedirs(config.get("tmp_dir", "tmp"), exist_ok=True)    
    print(f"[bold green]Using config[/bold green]: {config}")
    if crate is None:
      raise ValueError("[bold red]Crate must be provided[/bold red] when getting a task.")       
    config = {
      **config,
      "file_format": crate.dereference("./").get("file_format", "grapto").replace(".", "")     
    }       
    task, ValidEngine = TaskManager.get(task, config)           
    new_engine_path = task.train()       
    if new_engine_path:
      engine = ValidEngine(name=new_engine_path, config=config)        
      task.add_engine(engine, crate)
      print(f"[green]New engine added:[/green] {task.engine}")      
    print(f"[bold green]Training completed[/bold green]: {task.id}")    
    task_crate = task.to_crate(crate)              
    return task_crate

  @staticmethod
  def eval(task: str, config: dict = {}):
    crate: ROCrate = config.get("crate")
    os.makedirs(config.get("tmp_dir", "tmp"), exist_ok=True)
    print(f"[bold green]Using config[/bold green]: {config}")
    if crate is None:
      raise ValueError("[bold red]Crate must be provided[/bold red] when getting a task.")
    config = {
      **config,
      "file_format": crate.dereference("./").get("file_format", "grapto").replace(".", "")
    }
    task, _ = TaskManager.get(task, config)
    task.eval()        

  @staticmethod
  def cluster(task: str, config: dict = {}):
    crate: ROCrate = config.get("crate")
    os.makedirs(config.get("tmp_dir", "tmp"), exist_ok=True)
    print(f"[bold green]Using config[/bold green]: {config}")
    if crate is None:
      raise ValueError("[bold red]Crate must be provided[/bold red] when getting a task.")
    config = {
      **config,
      "file_format": crate.dereference("./").get("file_format", "grapto").replace(".", "")
    }
    task, _ = TaskManager.get(task, config)
    task.cluster()    

  