from __future__ import annotations
import os
from typing import List
from pathlib import Path
from rich import print
from rocrate.rocrate import ROCrate
from rocrate.model.contextentity import ContextEntity
from .utils import FileType, BaseROCrate, Task, generate_new_crate  
from ..task.models import Task, TaskModel

class ImageFile(ContextEntity):
    """
    Represents an image file in the Collectra workflow.
    This class extends ContextEntity to include properties specific to image files.
    """
    def __init__(self, name: str, version: str, output: Path):
        super().__init__(name, version, output)
        print(f"ImageFile initialized: {self.name} version {self.version}")

class Collectra(BaseROCrate):    
    def __init__(self, name: str, version: str, output: Path, crate: ROCrate = None):        
        super().__init__(name, version, output, crate)        
        print(f"Collectra workflow initialized: {self.name} version {self.version}")

    def generate_crate(self, obj: BaseROCrate, type: FileType, tasks: List[Task] = []) -> ROCrate:
        """
        Generate a ROCrate for the Collectra workflow.
        This method creates the necessary directories and files for the workflow.
        """
        os.makedirs(self.output, exist_ok=True)        
        crate = ROCrate()
        file_name = self.output / f"{self.name}.{type.value}"        
        for task in tasks:
            task = TaskModel.get_task(task) 
            task_crate = Task(crate, task.task_type, properties = {
                "description": task.describe(),
                "task_type": task.task_type,                
            })           
            crate.add(task_crate)
        crate.write(file_name)                
        self.crate = crate
    
    def run(self):
        task_chain = []        
        for e in self.crate.get_entities():
            if e.type == "Task":
                task = TaskModel.build_task(e)                
                task_chain.append(task)
        print(f"[purple]Built task chain[/purple]: ", task_chain)
        for task in task_chain:
            task.run()

    @staticmethod
    def create(
        name="default",
        version="1.0",
        output: Path = Path.cwd(),
        tasks: List[str] = None
    ):
        """
        Create a Collectra workflow with the specified name, version, output directory, and tasks.
        If no tasks are provided, an empty workflow is created.
        """
        collectra = Collectra(name, version, output)        
        os.makedirs(collectra.output, exist_ok=True)                              
        return collectra.generate_crate(collectra, FileType.COLLECTRA, tasks)        

    @staticmethod
    def load_workflow(
        workflow_file: Path
    ) -> Collectra:
        collectra = Collectra(
            name=workflow_file.stem,
            version="1.0",
            output=workflow_file.parent,
            crate = ROCrate(workflow_file)
        )                      
        return collectra

class Grapto(BaseROCrate):  
    def __init__(self, name: str, version: str, output: Path):        
        super().__init__(name, version, output)
        print(f"Grapto initialized: {self.name} version {self.version}")
          
    @staticmethod
    def generate_grapto(
        name: str,
        version: str,
        output: Path = None
    ):
        grapto = Grapto(name, version, output)                     
        os.makedirs(grapto.output, exist_ok=True)        
        return generate_new_crate(grapto, FileType.GRAPTO)