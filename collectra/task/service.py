from .base import Task, TaskDefinition, TaskEntity, TaskType
from .detect_object import DetectObject
from typing import List
from rocrate.rocrate import ROCrate
from enum import Enum

TaskType: dict = {
  "detect_object": DetectObject,
  "classify_image": ClassifyImageTask,
  "ocr": OCRTask
}

class TaskType(Enum):
  DETECT_OBJECT = "detect_object"
  CLASSIFY_IMAGE = "classify_image"
  OCR = "ocr"
  ADJUST_DATABASE = "adjust_database"
  LLM_CORRECTOR = "llm_corrector"

class EngineManager:    
  pass

class TaskManager:  

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
    return list(TaskManager.TASK_DEFINITIONS.keys())
  
  def chain(self, task: TaskEntity, config: dict = {}) -> Task:
    if not task:
      return None         
    task_type = task["task_type"]
    engine = task["engine"][0]
    engine_model = EngineManager.get(engine["name"])
    task = Task(TaskManager.TASK_DEFINITIONS[task_type], config=config)
    task.add_engine(engine_model)
    self.task_chain.append(task)
    return task    
  
  def get(task: str, crate: ROCrate = None) -> Task:
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
    if task_type not in TaskManager.TASK_DEFINITIONS:
      raise ValueError(f"[bold red]Invalid task type[/bold red]: {task_type}. Must be one of {list(TASK_DEFINITIONS.keys())}.")
    engine = EngineManager.get(engine)
    task: Task = Task(TaskManager.TASK_DEFINITIONS[task_type])
    task.add_engine(engine)
    print(engine)
    task.to_crate(crate)
    return task
    
  