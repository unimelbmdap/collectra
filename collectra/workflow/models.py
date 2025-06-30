import os
from pathlib import Path
from rich import print
from rocrate.rocrate import ROCrate

class BaseROCrate:
    def __init__(self, name: str, version: str, output: Path):
        self.name = name
        self.version = version
        self.output = output        

class Collectra(BaseROCrate):    
    def __init__(self, name: str, version: str, output: Path):        
        super().__init__(name, version, output)
        print(f"Collectra workflow initialized: {self.name} version {self.version}")

    @staticmethod
    def build(
        name: str,
        version: str,
        output: Path = None
    ) -> ROCrate:
        workflow = Collectra(name, version, output)                  
        os.makedirs(workflow.output, exist_ok=True)
        crate = ROCrate()
        file_name = workflow.output / f"{workflow.name}.collectra"        
        crate.write_zip(file_name)
        return crate

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
        crate = ROCrate()
        file_name = grapto.output / f"{grapto.name}.grapto"        
        crate.write(file_name)
        return crate