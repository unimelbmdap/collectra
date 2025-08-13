from __future__ import annotations
from abc import ABC, abstractmethod
import uuid
from typing import Tuple
from rich import print
from rocrate.model.data_entity import DataEntity
from rocrate.rocrate import ROCrate
from .model import *

#---------------------------------------------

class TaskEntity(DataEntity):
  def __init__(self, crate, identifier=None, properties=None):
    super(TaskEntity, self).__init__(crate, identifier, properties)
  
  def _empty(self):
    return {
        "@id": self.id,
        "@type": "Task",                     
        "task_type": "",        
    } 
  
class Task(ABC):

  VALID_MODELS: dict = {}
  VALID_INPUTS = []
  VALID_OUTPUTS = []

  def __init__(self, id: str = None, model_path: str = None, model_type: str = None, input: list = [], output: list = [], config: dict = {}):
    self.id = id if id else uuid.uuid4()            
    self.model = self.validate_model(model_path, model_type)   
    self.old_model = None         
    self.input = input if len(input) > 0 else self.VALID_INPUTS
    self.output = output if len(output) > 0 else self.VALID_OUTPUTS        
    self.config = config

  def get_metadata(self) -> dict:
    """
    Get metadata of the task.
    :return: A dictionary containing task metadata.
    """
    return {
        "id": self.id,        
        "model": self.model.name if self.model else None,
        "config": self.config,
    }
  
  def __str__(self) -> str:
    return f"{self.id} of {self.__class__.__name__}"  

  def __repr__(self) -> str:
    return f"{self.id} of {self.__class__.__name__}"  

  def to_crate(self, crate: ROCrate) -> ROCrate:
    """
    Convert the TaskModel to a ROCrate entity.
    :param crate: The ROCrate instance to add the task to.
    :return: The created Task entity.
    """    
    task_crate: TaskEntity = TaskEntity(crate, identifier=self.id, properties={        
        "task_type": self.task_type,                
    })     
    if self.model:   
      task_crate["engine"] = [self.model.to_crate(crate)]
    crate.add(task_crate)
    return task_crate

  def validate_model(self, model_path: str, model_type: str) -> Model:  
    print(f"Validating model with path {model_path} and type {model_type}")  
    if model_type not in self.VALID_MODELS:
        raise Exception(f"[bold red]Invalid model type[/bold red]: {model_type}. Must be one of {list(self.VALID_MODELS.keys())}.")
    model = self.VALID_MODELS.get(model_type)(model_path)
    return model
  
  def slug(self) -> str:
    return self.task_type

  def get_model(self) -> Model:
    """
    Get the model associated with the task.
    :return: The Engine instance if set, otherwise None.
    """
    return self.model

  def add_model(self, model: Model) -> bool:
    """
    Add a model to the task.
    :param model: The EngineModel instance to add.
    """ 
    validated = self.validate_model(model)            
    return validated

  def delete(self):
    if isinstance(self.model, Model):
      self.model.delete()

  def set_config(self, config: dict) -> None:
    """
    Set the configuration for the task.
    :param config: A dictionary containing configuration parameters.
    """
    if not isinstance(config, dict):
      raise ValueError("Config must be a dictionary.")
    self.config.update(config)

  @abstractmethod
  def run(self) -> None:
    pass

  @abstractmethod
  def train(self) -> None:
    pass

  @abstractmethod
  def eval(self) -> None:
    pass
  

class ObjectDetectionYOLO(Task):

  VALID_MODELS: dict = {
      "yolo": YOLOModel,
      "detectron2": DETECTRON2Engine,
  }      

  VALID_INPUTS =  ["specimen_sheet"]
  VALID_OUTPUTS = ["primary_specimen_label", "handwritten_data",
                   "annotation_label", "stamp", "swing_tag",
                   "accession_number", "small_database_label",
                   "medium_database_label", "full_database_label",
                   "swatch", "scale", "institutional label", "swing tag", "annotation label", "handwritten data", "number"]

  def run(self) -> None:    
    if not self.model:
      raise ValueError("Model must be set before running the task.")
    return self.model.detect(self.config)
  
  def train(self) -> None:
    if not self.model:
      raise ValueError("Engine must be set before training the task.")
    print(self.config)
    return self.model.train(self.config)
  
  def eval(self) -> None:
    if not self.model:
      raise ValueError("Engine must be set before validating the task.")
    self.model.val(self.config)
  
  def cluster(self) -> None:
    if not self.model:
      raise ValueError("Engine must be set before clustering the task.")
    self.model.cluster(self.config)

class TextClassification(Task):

  VALID_MODELS: Tuple = (
      ImageClassifier,
  )

  def __init__(self, task_type: str = "classify_image", config: dict = {}, model: Model = None):
    super().__init__(task_type, config, model)
  
  def run(self) -> None:
    if not self.model:
      raise ValueError("Engine must be set before running the task.")
    self.model.run(self.config)
  
  def train(self) -> Path:
    if not self.model:
      raise ValueError("Engine must be set before training the task.")
    return self.model.train(self.config)


  def eval(self) -> None:
    if not self.model:
      raise ValueError("Engine must be set before validating the task.")
    self.model.val(self.config)
        
  



  

  

