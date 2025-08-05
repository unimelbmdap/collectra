from __future__ import annotations
from rich import print
from abc import ABC, abstractmethod
from pathlib import Path
from rocrate.rocrate import ROCrate
from rocrate.model import DataEntity
from ultralytics import YOLO
from tqdm import tqdm

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
  
  DEFAULT_CONFIG: dict = {}

  def __init__(self,
      name: str | Path,                        
      config: dict = {}
    ):
    self.name: Path = Path(name)            
    self.model = None
    self.type = "generic"

  def __str__(self) -> str:
    return f"{self.name}"  

  @abstractmethod
  def to_crate(self, crate: ROCrate) -> EngineEntity:    
    pass
  
  @abstractmethod
  def train(self, config: dict = {}) -> None:
    pass

  @abstractmethod
  def validate(self) -> None:
    pass

  @abstractmethod
  def detect(self, data: Path) -> None:
    pass

  @abstractmethod
  def preprocess(self, data: Path, file_format: str) -> None:
    pass

class YOLOEngine(Engine):

  DEFAULT_CONFIG: dict = {    
    "epochs": 50,
    "imgsz": 640,    
    "verbose": True,
    "device": "cpu"
  }

  def __init__(self, name: str | Path = "yolo11n.pt"):
    super().__init__(name)
    self.model: YOLO = YOLO(self.name, verbose=True)
    self.type: str = "yolo"
    self.yolo_config_path: str = "yolo_config.yml"

  def preprocess(self, data: Path, file_format: str):
    train_files = []
    val_files = []
    number_of_classes = 0
    classes = []    
    try:
      self.dir
    except AttributeError:
      self.dir = Path("tmp")    
    for file in tqdm(data.glob(f"*.{file_format}")):
      try:
        crate = ROCrate(file)
        if number_of_classes == 0:        
          for e in crate.data_entities:
            if e.type == "ClassMapping":
              print(f"[bold green]Found class mapping[/bold green]: {e.get('classes')}")
              classes = e.get("classes") or []
              number_of_classes = len(classes)                    
              break                      
        
        bounding_box = ""        
        for e in crate.data_entities:
          if e.type == "File":
            e.write(Path(self.dir))          
            if e.get("for_validation"):              
              val_files.append(f"./{e.id}")
            else:
              train_files.append(f"./{e.id}")        
          if e.type == "BoundingBox":
            bounding_box += f"{e.get('class_id')} {e.get('x_center')} {e.get('y_center')} {e.get('width_relative')} {e.get('height_relative')}\n"        
        bounding_box_file = Path(self.dir) / f"{file.stem}.txt"
        bounding_box_file.write_text(bounding_box)                             
      except Exception as e:
        print(f"[bold red]Error processing file[/bold red]: {file} - {e}")
    yolo_config = f"train: train.txt\nval: val.txt\nnc: {number_of_classes}\nnames: {classes}"    
    Path(f"{self.dir}/train.txt").write_text("\n".join(train_files))
    Path(f"{self.dir}/val.txt").write_text("\n".join(val_files))    
    Path(f"{self.dir}/{self.yolo_config_path}").write_text(yolo_config)

  def train(self, config: dict = {}) -> Path:
    data=Path(config.get("input", "data"))      
    file_format = config.get("file_format", "grapto").replace(".", "")
    self.dir = Path(config.get("tmp_dir", "tmp"))
    print(f"[bold green]Training object detection model[/bold green]: {self.name}")    
    self.preprocess(data, file_format=file_format)   
    print(f"{self.dir}/{self.yolo_config_path}")                 
    train_results = self.model.train(
      data=Path(f"{self.dir}/{self.yolo_config_path}"),
      epochs=config.get("epochs", self.DEFAULT_CONFIG["epochs"]),
      imgsz=config.get("imgsz", self.DEFAULT_CONFIG["imgsz"]),
      # device=config.get("device", self.DEFAULT_CONFIG["device"]),      
      verbose=config.get("verbose", self.DEFAULT_CONFIG["verbose"]),   
      project=self.dir  
    )            
    new_model_path = Path(self.dir) / "train" / "weights" / "best.pt"    
    metrics = self.model.val()     
    logs = Path(config.get("output", Path.cwd()) / "logs.txt")
    logs.parent.mkdir(parents=True, exist_ok=True)
    logs.write_text(f"Training results: {train_results} \n Validation metrics: {metrics}")
    return new_model_path      
    
  def to_crate(self, crate: ROCrate) -> EngineEntity:
    """
    Convert the EngineModel to a ROCrate entity.
    :param crate: The ROCrate instance to add the engine to.
    :return: The created Engine entity.
    """            
    model_path = Path(self.name)
    if not model_path.exists():
      raise ValueError(f"[bold red]Model file does not exist[/bold red]: {model_path}")
    engine_crate = crate.add_file(model_path, properties={
      "name": f"{model_path}",    
      "engine_type": self.type  
    })
    engine_crate["name"] = engine_crate.id
    training_params = TrainingParameters(crate, identifier=f"{engine_crate.id}-training-params", properties=self.DEFAULT_CONFIG)               
    crate.add(engine_crate)     
    crate.add(training_params)
    engine_crate["trainingParameters"] = [training_params]
    return engine_crate
  
  def validate(self):
    print(f"[bold green]Validating object detection model[/bold green]: {self.name}")
    # Implement validation logic here
    pass

  def detect(self):
    print(f"[bold green]Running object detection[/bold green]: {self.name}")    
    # Implement logic to handle results
    pass

class DETECTRON2Engine(Engine):
    def __init__(self, name: str | Path = "detectron2.pt"):
      super().__init__(name)
      self.type = "detectron2"


class ImageClassifier(Engine):
    """
    A simple image classifier engine.
    """
    def __init__(self, name: str | Path = "imageclassifier.pt"):
      super().__init__(name)
      self.type = "image_classifier"

    def detect(self, config: dict = {}):
      print(f"[bold green]Running image classifier[/bold green]: {self.name}")
      # Implement the logic to classify images
      pass
    
    def train(self, config: dict = {}):
      print(f"[bold green]Training image classifier[/bold green]: {self.name}")
      # Implement the logic to train the image classifier
      pass
    
    def validate(self):
      print(f"[bold green]Validating image classifier[/bold green]: {self.name}")
      # Implement the logic to validate the image classifier
      pass

    def to_crate(self, crate: ROCrate) -> EngineEntity:
      pass

    def preprocess(self, data, file_format):
      pass

