from __future__ import annotations
import os
from shutil import rmtree
from typing import List
from pathlib import Path
from rich import print
from rocrate.rocrate import ROCrate
from rocrate.model.contextentity import ContextEntity
from .utils import FileType, BaseROCrate, Task, make_default_crate  
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
    def __init__(self, name: str, version: str, output: Path, crate: ROCrate = ROCrate()):        
        super().__init__(name, version, output, crate)        
        print(f"Collectra workflow initialized: {self.name} version {self.version}")
        self.file_name = self.output / f"{self.name}.{FileType.COLLECTRA.value}"
        self.task_chain = []        
    
    def run(self, input: Path, output: Path):        
        config = {
            "input": input,
            "output": output
        }
        for e in self.crate.get_entities():
            if e.type == "Task":                
                task = TaskModel.build_task(e, config)                
                self.task_chain.append(task)
        print(f"[purple]Built task chain[/purple]: ", self.task_chain)
        for task in self.task_chain:
            task.run()

    def add_task(self, task: str):
        try:
            task = TaskModel.get_task(task)
            task_crate = Task(self.crate, task.task_type, properties = {
                "description": task.describe(),
                "task_type": task.task_type,                
            })
            self.crate.add(task_crate)
        except ValueError as e:
            print(f"[red]Error adding task[/red]: {e}")
            return
    
    def add_tasks(self, tasks: List[str]):
        """
        Add multiple tasks to the Collectra workflow.
        :param tasks: List of task definitions in the format "<task_type>,<engine>".
        """
        for task in tasks:
            self.add_task(task)
    
    def save(self, compressed: bool = True):
        """
        Save the current state of the Collectra workflow to the output directory.
        """
        if compressed:
            self.crate.write_zip(self.file_name)
            return        
        self.crate.write(self.file_name)
    
    def delete(self, task: str):
        """
        Remove a task from the Collectra workflow.
        :param task: The ID of the task to remove.
        """
        try:
            print(f"[purple]Attempting to remove task[/purple]: {task}")
            task = self.crate.dereference(task)                  
            self.crate.delete(task)
            print(f"[green]Task removed successfully.[/green]")
        except ValueError as e:
            print(f"[red]Error removing task[/red]: {e}")

    @staticmethod
    def make(
        name="default",
        version="1.0",
        output: Path = Path.cwd(),
        tasks: List[str] = None
    ) -> Collectra:
        """
        Create a Collectra workflow with the specified name, version, output directory, and tasks.
        If no tasks are provided, an empty workflow is created.
        """
        print(f"Creating workflow: {name} in {output}")
        workflow_p = Path(f"{output}/{name}.collectra")
        if workflow_p.exists() and name != "default":
            print(f"[red]Workflow file already exists[/red]: {name} - skipping initialisation. To edit use `collectra edit` command.")
            return            
        if name == "default":
            rmtree(workflow_p, ignore_errors=True)
        os.makedirs(output, exist_ok=True)                
        collectra = Collectra(name, version, output)                                              
        collectra.add_tasks(tasks)    
        collectra.save()              
        return collectra        

    @staticmethod
    def load_workflow(
        workflow_file: Path
    ) -> Collectra:
        if not workflow_file.exists():
            print(f"Error: The specified workflow does not exist: {workflow_file}")
            return
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
        return make_default_crate(grapto, FileType.GRAPTO)