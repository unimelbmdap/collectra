import os
from pathlib import Path
from ultralytics import YOLO

class CollectraTaskManager:
  @staticmethod
  def create_temp_workdir(dir_name: Path):
    os.makedirs(dir_name, exist_ok=True)

class TrainObjectDetection:
  
  def __init__(self, model_name: str | Path = "yolo11n.pt", output: Path = Path("output")):    
    self.model = model_name
    self.output = output    

  def run(self):
    CollectraTaskManager.create_temp_workdir(self.output)
    model_path = self.output / "model"
    train_path = self.output / "run"
    model = YOLO(model_path / self.model)
    train_results = model.train(
      data=Path("glabels/yolo_config.yml"),
      epochs=50,
      imgsz=640,
      device="0",
      verbose=True,
      project=train_path
    )
    metrics = model.val() 

trainer = TrainObjectDetection(model_name="yolo11n.pt")
trainer.run()   