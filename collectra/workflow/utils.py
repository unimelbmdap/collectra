import os
from typing import List
from enum import StrEnum
from rich import print
from rocrate.rocrate import ROCrate
from rocrate.model.contextentity import ContextEntity
from pathlib import Path

class TaskType(StrEnum):
    ANNOTATION_CONVERSION = "convert_annotation"
    OBJECT_DETECT = "detect_objects"
    LABEL_CLASSIFY = "classify_label"
    CHAR_RECOGNISE = "extract_text"        

class FileType(StrEnum):
    COLLECTRA = "collectra"
    GRAPTO = "grapto"

class Engine(StrEnum):
    YOLO = "yolo2"
    DETECTRON = "detectron2"
    TESSERACT = "tesseract"
    INTERNAL = ""

class Task:
    task_type: TaskType
    engine: Engine

    def __init__(self, task_type: TaskType, engine: Engine):
        self.task_type = task_type
        self.engine = engine    

    def get_engine(self) -> Engine:
        return self.engine 

    def get_task_type(self) -> TaskType:
        return self.task_type

class BaseROCrate:
    def __init__(self, name: str, version: str, output: Path, crate: ROCrate = None):
        self.name = name
        self.version = version
        self.output = output
        self.crate = crate     

def verify_tasks(tasks: List[str]) -> List[Task]:
    verified_tasks: List[Task] = []
    for task in tasks:        
        try:            
            task_type = task
            engine = Engine.INTERNAL
            if "," in task:
                task_type, engine = task.split(",")                        
            if task_type not in iter(TaskType):
                raise ValueError(f"[bold red]Invalid task type[/bold red]: {task_type}. Must be one of {list(TaskType)}.")
            if engine not in iter(Engine):
                raise ValueError(f"[bold red]Invalid engine[/bold red]: {engine}. Must be one of {list(Engine)}.")
            verified_tasks.append(Task(TaskType(task_type), Engine(engine)))    
            print(f"[bold green]Verified task[/bold green]: [bold purple]{task_type}[/bold purple] with engine: [bold purple]{engine if engine else 'default'}[/bold purple]")           
        except Exception as e:
            print(f"Error verifying task '{task}': {e} - Skipping...")            
            continue        
    return verified_tasks

def generate_new_crate(obj: BaseROCrate, type: FileType) -> ROCrate:
    os.makedirs(obj.output, exist_ok=True)        
    crate = ROCrate()
    file_name = obj.output / f"{obj.name}.{type.value}"        
    crate.write_zip(file_name)
    return crate