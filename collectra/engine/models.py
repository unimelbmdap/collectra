from pathlib import Path
from rocrate.rocrate import ROCrate
import os, shutil, random
from rich import print
from ultralytics import YOLO
from rocrate.model.contextentity import ContextEntity

class Engine(ContextEntity):
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

class TrainingParameters(ContextEntity):
  def __init__(self, crate, identifier=None, properties=None):
    super(TrainingParameters, self).__init__(crate, identifier, properties)
  
  def _empty(self):
    return {
        "@id": self.id,
        "@type": "TrainingParameters",        
    }

class EngineRepo:
  pass
    
class TrainObjectDetection:
  
  def __init__(self, data: Path, model_name: str | Path = "yolo11n.pt", output: Path = Path("output")):    
    self.data = data
    self.model = model_name
    self.output = output    

  def run(self):
    print(f"[bold green]Training object detection model[/bold green]: {self.model}")
    os.makedirs("tmp", exist_ok=True)
    list_of_files = []    
    for file in self.data.glob("*.grapto"):
      crate = ROCrate(file)
      for e in crate.data_entities:
        e.write(Path("tmp"))
        list_of_files.append(f"./{file.stem}.jpg")                
      print(f"[bold green]Processing file[/bold green]: {file}")
      bounding_box = ""
      for e in crate.contextual_entities:
        if e.type == "ImageBoundingBox":
          bounding_box += f"{e.get("class_id")} {e.get("x_center")} {e.get("y_center")} {e.get("width_relative")} {e.get("height_relative")}\n"
      bounding_box_file = Path("tmp") / f"{file.stem}.txt"
      bounding_box_file.write_text(bounding_box)      

    number_of_classes = 0
    classes = []
    
    for file in self.data.glob("*.grapto"):
      crate = ROCrate(file)
      for e in crate.contextual_entities:
        if e.type == "ClassMapping":
          print(f"[bold green]Found class mapping[/bold green]: {e.get('classes')}")
          classes = e.get("classes") or []
          number_of_classes = len(classes)                    
          break        
      break

    yolo_config = f"train: train.txt\nval: val.txt\nnc: {number_of_classes}\nnames: {classes}"

    Path("tmp/yolo_config.yml").write_text(yolo_config)

    random.seed(448)
    random.shuffle(list_of_files)
    print(list_of_files)

    # Randomly split 80% for training and 20% for validation
    train_files = list_of_files[:int(len(list_of_files) * 0.8)]
    val_files = list_of_files[int(len(list_of_files) * 0.8):]

    Path("tmp/train.txt").write_text("\n".join(train_files))
    Path("tmp/val.txt").write_text("\n".join(val_files))  

    model = YOLO(self.model)
    train_results = model.train(
      data=Path("tmp/yolo_config.yml"),
      epochs=50,
      imgsz=640,
      device="cpu",
      verbose=True,      
    )
    metrics = model.val() 
    shutil.rmtree("tmp", ignore_errors=True)
