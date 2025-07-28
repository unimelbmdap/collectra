from __future__ import annotations
from pathlib import Path
from rocrate.rocrate import ROCrate
import os, shutil, random
from rich import print

from rocrate.model import DataEntity
from enum import Enum

class EngineEntity(DataEntity):
  def __init__(self, crate, identifier=None, properties=None):
    super(EngineEntity, self).__init__(crate, identifier, properties)
  
  def _empty(self):
    return {
        "@id": self.id,
        "@type": "Engine",
        "name": "",
        "version": "",
        "description": "",
        "source": "",        
    }  

class TrainingParameters(DataEntity):
  def __init__(self, crate, identifier=None, properties=None):
    super(TrainingParameters, self).__init__(crate, identifier, properties)
  
  def _empty(self):
    return {
        "@id": self.id,
        "@type": "TrainingParameters",                        
    }

class Engine:
  def __init__(self, name: str, version: str = "", source: str = "default", description: str = "", config: dict = {}):
    self.name: str = name
    self.version: str = version
    self.source: str = source
    self.description: str = description
    self.config: dict = config

  def __str__(self) -> str:
    return f"{self.name}"
  
  def describe(self) -> str:
    return f"{self.name} version {self.version} from {self.source}. {self.description}"

  def to_crate(self, crate: ROCrate) -> EngineEntity:
    """
    Convert the EngineModel to a ROCrate entity.
    :param crate: The ROCrate instance to add the engine to.
    :return: The created Engine entity.
    """
    engine_crate = EngineEntity(crate, identifier=f"{self.name}-{self.source}", properties={
        "name": self.name,
        "version": self.version,
        "source": self.source,
        "description": self.description,
    })    
    crate.add_file(None, f"{self.name}-{self.source}", properties={
      "name": f"{self.name}-{self.source}",
      "description": self.description,
      "version": self.version,
      "source": self.source,
    })
    # crate.add(engine_crate)     
    return engine_crate
  
  def run(self, config: dict = {}) -> None:
    print("Found config: ", config)
    print(f"[bold green]Running engine[/bold green]: {self.name} source {self.source}")    
    pass

class EngineDefinition:

  def __init__(self, name: str, engine: Engine):
    self.name: str = name        
    self.engine: Engine = engine    
  
  def __str__(self) -> str:
    return f"{self.name} from {self.engine.source} version {self.engine.version}"   
  
  def get(self) -> Engine:
    return self.engine

