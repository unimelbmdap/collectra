from __future__ import annotations
from pathlib import Path
from rocrate.rocrate import ROCrate
import os, shutil, random
from rich import print
from ultralytics import YOLO
from rocrate.model import DataEntity
from grapto.main import run as run_grapto
from urllib.parse import unquote

class Engine(DataEntity):
  def __init__(self, crate, identifier=None, properties=None):
    super(Engine, self).__init__(crate, identifier, properties)
  
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

class BaseEngine:
  def __init__(self, name: str, version: str = "", source: str = "default", description: str = "", config: dict = {}):
    self.name = name
    self.version = version
    self.source = source
    self.description = description
    self.config = config

  def __str__(self):
    return f"{self.name}"
  
  def describe(self):
    return f"{self.name} version {self.version} from {self.source}. {self.description}"

  def to_crate(self, crate: ROCrate) -> Engine:
    """
    Convert the EngineModel to a ROCrate entity.
    :param crate: The ROCrate instance to add the engine to.
    :return: The created Engine entity.
    """
    engine_crate = Engine(crate, identifier=f"{self.name}-{self.source}", properties={
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
  
  def run(self, config: dict = {}):
    print("Found config: ", config)
    print(f"[bold green]Running engine[/bold green]: {self.name} source {self.source}")    
    pass
    
class YoloEngine(BaseEngine):
  def __init__(self, name: str | Path = "yolo11n.pt"):  
    super(YoloEngine, self).__init__(name)       
    self.dir = Path("tmp")   

  def run(self, config: dict = {}):
    data=Path(config.get("output"))    
    print(f"[bold green]Training object detection model[/bold green]: {self.name}")
    os.makedirs(self.dir, exist_ok=True)
    list_of_files = []    
    for file in data.glob("*.grapto"):
      crate = ROCrate(file)
      for e in crate.data_entities:
        e.write(Path(self.dir))
        list_of_files.append(f"./{file.stem}.jpg")                
      print(f"[bold green]Processing file[/bold green]: {file}")
      bounding_box = ""
      for e in crate.contextual_entities:
        if e.type == "ImageBoundingBox":
          bounding_box += f"{e.get("class_id")} {e.get("x_center")} {e.get("y_center")} {e.get("width_relative")} {e.get("height_relative")}\n"
      bounding_box_file = Path(self.dir) / f"{file.stem}.txt"
      bounding_box_file.write_text(bounding_box)      

    number_of_classes = 0
    classes = []
    
    for file in data.glob("*.grapto"):
      crate = ROCrate(file)
      for e in crate.contextual_entities:
        if e.type == "ClassMapping":
          print(f"[bold green]Found class mapping[/bold green]: {e.get('classes')}")
          classes = e.get("classes") or []
          number_of_classes = len(classes)                    
          break        
      break

    yolo_config = f"train: train.txt\nval: val.txt\nnc: {number_of_classes}\nnames: {classes}"

    Path(f"{self.dir}/yolo_config.yml").write_text(yolo_config)

    random.seed(448)
    random.shuffle(list_of_files)    

    # Randomly split 80% for training and 20% for validation
    train_files = list_of_files[:int(len(list_of_files) * 0.8)]
    val_files = list_of_files[int(len(list_of_files) * 0.8):]

    Path("tmp/train.txt").write_text("\n".join(train_files))
    Path("tmp/val.txt").write_text("\n".join(val_files))  

    model = YOLO(self.name)
    train_results = model.train(
      data=Path("tmp/yolo_config.yml"),
      epochs=3,
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

class VIAEngine(BaseEngine):
  
  def __init__(self, name: str = "via"):
    super(VIAEngine, self).__init__(name)

  def run(self, config: dict = {}):
    path = Path(config.get("input"))
    output = Path(config.get("output"))
    function = config.get("engine", "via")
    run_grapto(
      path=path,
      output=output,
      function=function
    )

class EngineManager:

  ENGINE_DEFINITIONS: dict = {
    "yolo11n.pt": YoloEngine("yolo11n.pt"),
    "via": VIAEngine("via"),
  }
  
  @staticmethod
  def get(name: str) -> BaseEngine:
    """
    Get an engine by name.
    :param name: The name of the engine.
    :return: An instance of BaseEngine.
    """
    if name not in EngineManager.ENGINE_DEFINITIONS:
      raise ValueError(f"Unknown engine: {name}")
    return EngineManager.ENGINE_DEFINITIONS[name]

  @staticmethod
  def build(engine: str):
    if engine not in EngineManager.ENGINE_DEFINITIONS:
      raise ValueError(f"Unknown engine: {engine}")
    return EngineManager.ENGINE_DEFINITIONS[engine]    