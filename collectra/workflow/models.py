import os
from typing import List
from pathlib import Path
from rich import print
from rocrate.rocrate import ROCrate
from .utils import FileType, BaseROCrate, Task, verify_tasks, generate_new_crate  

class Collectra(BaseROCrate):    
    def __init__(self, name: str, version: str, output: Path):        
        super().__init__(name, version, output)
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
            task_type = task.get_task_type()
            engine = task.get_engine()
            crate.add_file("yolo/engine", dest_path="model/", properties={
                "task_type": f"{task_type.value}",
                "engine": f"{engine.value}"
            })
        crate.write(file_name)                
        return crate
    
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
        tasks = verify_tasks(tasks)              
        return collectra.generate_crate(collectra, FileType.COLLECTRA, tasks=tasks)    

    @staticmethod
    def add_to_workflow(
        workflow: ROCrate,
        name: str,
        version: str,
        output: Path = None
    ) -> ROCrate:
        collectra = Collectra(name, version, output)        
        return generate_new_crate(collectra, FileType.COLLECTRA, workflow)  

    @staticmethod
    def load_workflow(
        workflow_file: Path
    ) -> ROCrate:
        return ROCrate(workflow_file)           

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