from __future__ import annotations
import os, shutil
from pathlib import Path
from rich import print
from grapto.main import run as run_grapto
from rocrate.model.contextentity import ContextEntity
from rocrate.rocrate import ROCrate
from ultralytics import YOLO
from PIL import Image, ImageOps

# TODO: Move this to a task library module ----

class TaskRepo:    
  @staticmethod
  def convert_annotation(config: dict = {}):
    path = Path(config.get("input"))
    output = Path(config.get("output"))
    function = config.get("engine", "via")
    run_grapto(
      path=path,
      output=output,
      function=function
    )
  
  @staticmethod
  def detect_objects(config: dict = {}):
    engine = TrainObjectDetection(
      data= Path(config.get("input")),
      model_name=config.get("engine", "yolo11n.pt"),
      output=Path(config.get("output")),
    )
    engine.run()     

class TaskTypeChoice:
  def __init__(self, task_type: str, description: str, function: callable):
    self.task_type = task_type
    self.description = description
    self.function = function
  
  def run(self, configs: dict = {}):
    self.function(configs)
  
  def __str__(self):
    return f"{self.task_type}: {self.description}"  

TASK_TYPE_CHOICES: dict = {
    "convert_annotation": TaskTypeChoice(
      task_type="convert_annotation",
      description="Convert annotations from one format to another",
      function=TaskRepo.convert_annotation
    ),
    "detect_objects": TaskTypeChoice(
      task_type="detect_objects",
      description="Detect objects in images using a specified engine",
      function=TaskRepo.detect_objects
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
        "engine": "",        
        "task_type": "",        
    } 

class TaskModel:
  def __init__(self, task_type: str, task_type_choice: TaskTypeChoice, task: Task = None, config: dict = {}):
    self.task_type = task_type
    self.task_type_choice = task_type_choice
    self.config = config
    self.task = task

  def __str__(self):
    return f"{self.task_type}"
  
  def engine(self):
    return self.config.get("engine")

  def describe(self):
    description = self.task_type_choice.description if self.task_type_choice.description else f"{self.task_type_choice.function.__name__}"
    return description
  
  def run(self):        
    configuration_print = f"""
      [bold green]---[/bold green]
      Running task: {self.task_type}
      Engine: {self.engine()}
      Input: {self.config.get("input")}
      Output: {self.config.get("output")}
    """      
    print(configuration_print)              
    self.task_type_choice.run(self.config)
  
  @staticmethod
  def get_task(task: str) -> TaskModel:          
    if "," not in task:
      raise ValueError(f"[bold red]Invalid task format[/bold red]: {task}. Must be in the format <task_type>,<engine>.")
    task_type, engine = task.split(",")
    if task_type not in TASK_TYPE_CHOICES:
      raise ValueError(f"[bold red]Invalid task type[/bold red]: {task_type}. Must be one of {list(TASK_TYPE_CHOICES.keys())}.")
    return TaskModel(task_type, TASK_TYPE_CHOICES[task_type])
  
  @staticmethod
  def build_task(task: Task, config: dict = {}) -> TaskModel:        
    task_type = task["task_type"]    
    return TaskModel(task_type, TASK_TYPE_CHOICES[task_type], task, config=config)

class TaskManager:
  @staticmethod
  def create_temp_workdir(dir_name: Path):
    os.makedirs(dir_name, exist_ok=True)

class TrainObjectDetection:
  
  def __init__(self, data: Path, model_name: str | Path = "yolo11n.pt", output: Path = Path("output")):    
    self.data = data
    self.model = model_name
    self.output = output    

  def run(self):
    os.makedirs("tmp", exist_ok=True)
    for file in self.data.glob("*.grapto"):
      crate = ROCrate(file)
      for e in crate.data_entities:
        print(type(e))
        
    # model = YOLO(model_path / self.model)
    # train_results = model.train(
    #   data=Path("glabels/yolo_config.yml"),
    #   epochs=50,
    #   imgsz=640,
    #   device="0",
    #   verbose=True,
    #   project=train_path
    # )
    # metrics = model.val() 
    shutil.rmtree("tmp", ignore_errors=True)