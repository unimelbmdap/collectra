from .base import Task, TaskType
from engine import Engine
from pathlib import Path
from rocrate.rocrate import ROCrate
import random, shutil, os
from ultralytics import YOLO

class YOLOEngine(Engine):

  def __init__(self, name: str | Path = "yolo11n.pt"):  
    super(YOLOEngine, self).__init__(name)       
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
    shutil.rmtree(self.dir, ignore_errors=True)

class DETECTRON2Engine(Engine):
    pass

class DetectObject(Task):

    VALID_ENGINES = (
        YOLOEngine,
        DETECTRON2Engine,
    )

    def __init__(self, engine: Engine):
        super().__init__()
        self.validate_engine(engine)        

    def validate_engine(self, engine: Engine):
        if not isinstance(engine, self.VALID_ENGINES):
            raise ValueError(f"Invalid engine type: {type(engine).__name__}. "
                             f"Valid engines are: {[e.__name__ for e in self.VALID_ENGINES]}.")
        self.engine = engine