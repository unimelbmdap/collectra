from abc import ABC, abstractmethod
import uuid
from typing import Dict, List
from rich import print
from pathlib import Path
from .model import Model, YOLOModel, ImageClassifier

class Task(ABC):

  def __init__(self, 
    id: str,     
    input: List[str] = [], 
    output: List[str] = [], 
    config: Dict = {} 
  ):
    self.id = id if id else uuid.uuid4()                       
    self.input = input 
    self.output = output
    self.config = config
  
  def metadata(self) -> Dict:
    """
    Get metadata of the task.
    :return: A dictionary containing task metadata.
    """
    return {
        "id": self.id,                
        "config": self.config,
    }
  
  def __str__(self) -> str:
    return f"{self.id} of {self.__class__.__name__}"  

  def __repr__(self) -> str:
    return f"{self.id} of {self.__class__.__name__}"      

  def set_config(self, config: Dict) -> None:
    """
    Set the configuration for the task.
    :param config: A dictionary containing configuration parameters.
    """
    if not isinstance(config, Dict):
      raise ValueError("Invalid config type.")
    self.config.update(config)  

  @abstractmethod
  def run(self) -> None:
    """
    Run the task.
    This method should be implemented by subclasses.
    """
    raise NotImplementedError("Subclasses must implement this method.")
    

class MachineLearningTask(Task):

  VALID_MODEL = None

  def __init__(self, 
    id: str, 
    model: str | Path,     
    input: List[str] = [], 
    output: List[str] = [], 
    config: Dict = {} 
  ):
    super().__init__(id, input, output, config)            
    self.old_model = None   
    self.model = self.load(model)    

  def metadata(self) -> Dict:
    """
    Get metadata of the task.
    :return: A dictionary containing task metadata.
    """
    return {
        "id": self.id,        
        "model": self.model.name if self.model else None,
        "config": self.config,
    }             
  
  def load(self, model: str | Path) -> Model:  
    if not self.VALID_MODEL or not model:
      raise Exception("No valid model type defined for this task or model path is empty.")
    return self.VALID_MODEL(model)      

  def get_model(self) -> Model | None:
    """
    Get the model associated with the task.
    :return: The Engine instance if set, otherwise None.
    """
    return self.model

  def add_model(self, model: Path) -> Model:
    """
    Add a model to the task.
    :param model: The EngineModel instance to add.
    """ 
    return self.load(model)                

  def delete_model(self) -> None:
    """
    Delete the model associated with the task.
    This method sets the model to None.
    """
    self.model = None
  
  def run(self) -> None:    
    if not self.model:
      raise ValueError("Model must be set before running the task.")
    return self.model.detect(self.config)
  
  def train(self) -> None:
    if not self.model:
      raise ValueError("Model must be set before training the task.")
    self.model.train(self.config)
  
  def eval(self) -> None:
    if not self.model:
      raise ValueError("Model must be set before validating the task.")
    self.model.val(self.config)
  
  def cluster(self) -> None:
    # if not self.model:
    #   raise ValueError("Model must be set before clustering the task.")
    # self.model.cluster(self.config)
    pass


class ObjectDetectionYOLO(MachineLearningTask):     
  VALID_MODEL = YOLOModel

class TextClassification(MachineLearningTask):
  VALID_MODEL = ImageClassifier
        
  



  

  

