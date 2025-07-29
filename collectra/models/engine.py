from __future__ import annotations
import os, shutil, random
from rich import print
from abc import ABC, abstractmethod
from pathlib import Path
from rocrate.rocrate import ROCrate
from rocrate.model import DataEntity
from ultralytics import YOLO

class EngineEntity(DataEntity):
  def __init__(self, crate, identifier=None, properties=None):
    super(EngineEntity, self).__init__(crate, identifier, properties)
  
  def _empty(self):
    return {
        "@id": self.id,
        "@type": "Engine",
        "name": "",            
    }  

class TrainingParameters(DataEntity):
  def __init__(self, crate, identifier=None, properties=None):
    super(TrainingParameters, self).__init__(crate, identifier, properties)
  
  def _empty(self):
    return {
        "@id": self.id,
        "@type": "TrainingParameters",                        
    }

class Engine(ABC):

  TEMPORARY_DIR = Path("tmp")

  def __init__(self,
      name: str | Path,                        
      config: dict = {}
    ):
    self.name: Path = Path(name)    
    self.config: dict = config
    self.dir = Engine.TEMPORARY_DIR

  def __str__(self) -> str:
    return f"{self.name}"  

  def to_crate(self, crate: ROCrate) -> EngineEntity:
    """
    Convert the EngineModel to a ROCrate entity.
    :param crate: The ROCrate instance to add the engine to.
    :return: The created Engine entity.
    """        
    engine_crate = crate.add_file(None, f"{self.name}", properties={
      "name": f"{self.name}",      
    })
    # crate.add(engine_crate)     
    return engine_crate
  
  @abstractmethod
  def train(self, config: dict = {}) -> None:
    pass

class YOLOEngine(Engine):

  def __init__(self, name: str | Path = "yolo11n.pt"):  
    super().__init__(name)              

  def preprocess(self, data: Path, file_format: str):
    train_files = []
    val_files = []
    number_of_classes = 0
    classes = []    
    for file in data.glob(f"*.{file_format}"):
      try:
        crate = ROCrate(file)
        if number_of_classes == 0:        
          for e in crate.data_entities:
            if e.type == "ClassMapping":
              print(f"[bold green]Found class mapping[/bold green]: {e.get('classes')}")
              classes = e.get("classes") or []
              number_of_classes = len(classes)                    
              break                      
        for e in crate.data_entities:
          if e.type == "File":
            e.write(Path(self.dir))
            if e.get("for_validation"):              
              val_files.append(f"./{file.stem}.jpg")
            else:
              train_files.append(f"./{file.stem}.jpg")
        print(f"[bold green]Processing file[/bold green]: {file}")
        bounding_box = ""
        for e in crate.contextual_entities:
          if e.type == "ImageBoundingBox":
            bounding_box += f"{e.get('class_id')} {e.get('x_center')} {e.get('y_center')} {e.get('width_relative')} {e.get('height_relative')}\n"
        bounding_box_file = Path(self.dir) / f"{file.stem}.txt"
        bounding_box_file.write_text(bounding_box)                             
      except Exception as e:
        print(f"[bold red]Error processing file[/bold red]: {file} - {e}")
    yolo_config = f"train: train.txt\nval: val.txt\nnc: {number_of_classes}\nnames: {classes}"    
    Path(f"{self.dir}/train.txt").write_text("\n".join(train_files))
    Path(f"{self.dir}/val.txt").write_text("\n".join(val_files))    
    Path(f"{self.dir}/yolo_config.yml").write_text(yolo_config)

  def train(self, config: dict = {}):
    data=Path(config.get("data", "data"))  
    file_format = config.get("file_format", "grapto").replace(".", "")
    print(f"[bold green]Training object detection model[/bold green]: {self.name}")
    os.makedirs(self.dir, exist_ok=True)    
    self.preprocess(data, file_format=file_format)            
    random.seed(448)  
    model = YOLO(self.name)
    train_results = model.train(
      data=Path("tmp/yolo_config.yml"),
      epochs=1,
      imgsz=640,
      device="cpu",
      verbose=True,    
      project=self.dir  
    )
    crate = config.get("crate")
    if crate:
      crate.add_file(Path(self.dir) / "train" / "weights" / "best.pt", properties = {
        "name": f"{self.name}-best.pt",
        "description": "Best model weights after training",        
      })
    metrics = model.val() 
    # shutil.rmtree(self.dir, ignore_errors=True)

class DETECTRON2Engine(Engine):
    pass

class ImageClassifier(Engine):
    """
    A simple image classifier engine.
    """
    def __init__(self, name: str = "image_classifier"):
        super(ImageClassifier, self).__init__(name)
        self.model = None  # Placeholder for the model

    def run(self, config: dict = {}):
        print(f"[bold green]Running image classifier[/bold green]: {self.name}")
        # Implement the logic to classify images
        pass

