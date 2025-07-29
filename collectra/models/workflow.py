from __future__ import annotations
import os
from shutil import rmtree
from typing import List
from pathlib import Path
from rich import print
from rocrate.rocrate import ROCrate
from .utils import BaseROCrate  
from .service import TaskManager
from .task import TaskEntity

class Collectra(BaseROCrate):    

    TEMPORARY_DIR = Path("tmp")

    def __init__(self, name: str, version: str, output: Path, file_format: str = None, crate: ROCrate = ROCrate()):        
        super().__init__(name, version, output, crate)        
        print(f"Collectra workflow initialized: {self.name} version {self.version}")
        self.file_name = self.output / f"{self.name}.collectra"         
        if file_format:
            self.crate.update_jsonld({
                "@id": "./",
                "file_format": file_format,
            })                 

    def run(self, input: Path, output: Path):        
        config = {
            "input": input,
            "output": output,
            "crate": self.crate,
        }
        manager = TaskManager()        
        for e in self.crate.get_entities():
            if e.type == "Task":                
                manager.chain(e, config)                
        manager.run()
    
    def train(self, task: str,  input: Path, output: Path):
        config = {
            "input": input,
            "output": output,
            "crate": self.crate,
            "tmp_dir": self.TEMPORARY_DIR,
        }
        TaskManager.train(task, config)          
        self.save(compressed=True)  
        self.cleanup()      

    def add_task(self, task: str):
        try:
            return TaskManager.build(task, self.crate)
        except ValueError as e:
            print(f"[red]Error adding task[/red]: {e}")
            return
    
    def add_tasks(self, tasks: List[str]) -> List[TaskEntity]:
        """
        Add multiple tasks to the Collectra workflow.
        :param tasks: List of task definitions in the format "<task_type>,<engine>".
        """
        task_crates = []
        for task in tasks:
            task_crates.append(self.add_task(task))
        return task_crates
    
    def save(self, compressed: bool = True):
        """
        Save the current state of the Collectra workflow to the output directory.
        """
        if compressed:            
            if os.path.exists(Path(self.file_name)):
                print(f"[bold red]Replacing existing workflow file[/bold red]: {self.file_name}")
                os.remove(self.file_name)
            self.crate.write_zip(self.file_name)
            return        
        self.crate.write(self.file_name)
    
    def delete_task(self, task: str):
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
    
    def cleanup(self, tasks: List[TaskEntity] = []):
        """
        Cleanup temporary files created during the workflow execution.
        :param tasks: List of Task instances to clean up.
        """
        print(f"[blue]Cleaning up temporary files...[/blue]")
        for task in tasks:
            if task.engine:
                model_path = Path(task.engine.name)
                if model_path.exists():
                    os.remove(model_path)
                    print(f"[green]Removed temporary engine file[/green]: {model_path}")
        if self.TEMPORARY_DIR.exists():
            rmtree(self.TEMPORARY_DIR, ignore_errors=True)
            print(f"[green]Removed temporary directory[/green]: {self.TEMPORARY_DIR}")

    @staticmethod
    def make(
        name="default",
        version="1.0",
        output: Path = Path.cwd(),
        tasks: List[str] = None,
        file_format: str = "grapto",
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
        collectra = Collectra(name, version, output, file_format=file_format)
        tasks = collectra.add_tasks(tasks)
        collectra.save()
        collectra.cleanup(tasks)        
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