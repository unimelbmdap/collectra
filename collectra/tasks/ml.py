from pathlib import Path
from collectra.commons import MetaClass
from collectra.images.base import Image, ImageCrop
from collectra.models.base import Model
from collectra.models.yolo import YOLOModel
from collectra.tasks.base import Task
from dataclasses import dataclass
from types import UnionType
import tempfile, zipfile, copy, importlib


@dataclass(kw_only=True)
class MachineLearningTask(Task):

    VALID_MODEL = None

    def load(self, model: str | Path) -> Model:
        if not self.VALID_MODEL or not model:
            raise Exception(
                "No valid model type defined for this task or model path is empty."
            )        
        return self.VALID_MODEL(model)    

    def run(self, **kwargs) -> dict:
        if not self.config.get("model", None):
            raise ValueError("Model must be set before running the task.")
        return self.load(self.config.get("model")).detect(**kwargs)  # type: ignore

    def train(self) -> Path | None:
        if not self.config.get("model", None):
            raise ValueError("Model must be set before training the task.")
        model = Path(self.config.get("model", ""))
        if not model:
            raise ValueError("Model path must be set before training the task.")            
        if self.config.get("as_dir", False):            
            return self.load(model).train(self.config)
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                with zipfile.ZipFile(model, "r") as zip_ref:
                    zip_ref.extract(member=model.name, path=tmpdirname)
                    model_path = Path(tmpdirname) / model.name
                    return self.load(model_path).train(self.config)  # type: ignore

    def eval(self) -> None:
        if not self.config.get("model", None):
            raise ValueError("Model must be set before validating the task.")
        self.load(self.config.get("model")).val(self.config)  # type: ignore

    def cluster(self) -> None:
        # if not self.model:
        #   raise ValueError("Model must be set before clustering the task.")
        # self.model.cluster(self.config)
        pass

    def set_model(self, model: str | Path | None) -> None:
        """
        Set the model associated with the task.
        :param model: The model path as a string or Path object.
        """
        if model:
            self.config["old_model"] = self.config.get("model", "")
            self.config["model"] = str(model)

    def get_model(self) -> str:
        """
        Get the model associated with the task.
        :return: The model path as a string.
        """
        return str(self.config.get("model", ""))

    def get_old_model(self) -> str:
        """
        Get the old model associated with the task.
        :return: The old model path as a string.
        """
        return str(self.config.get("old_model", ""))

@dataclass(kw_only=True)
class ObjectDetectionYOLO(MachineLearningTask):
    VALID_MODEL = YOLOModel

    def input_type(self) -> UnionType:
        return Image | ImageCrop
    
    def output_type(self) -> type:
        return ImageCrop

    def check_kwargs(self, **kwargs) -> None:
        super().check_kwargs(**kwargs)
        if self.name == 'field_detector':
            print("Checking kwargs for field_detector")            
        for key in self.input:
            value = copy.deepcopy(self.input[key])
            if isinstance(value, dict) and "path" in value and "type" in value:                
                module_name, class_name = value.pop("type").rsplit(".", 1)
                cls = getattr(importlib.import_module(module_name), class_name)     
                path = value.pop("path")                                
                if "items" in value:                    
                    self.input[key] = []
                    for item in value["items"]:
                        self.input[key].append(cls.build(path, **item))
                else:
                    self.input[key] = cls.build(path, **value)            
            elif isinstance(value, str) and Path(value).is_file() and Image.is_image_file(Path(str(value))):
                self.input[key] = Image.build(Path(str(value)))                  

    def run(self, **kwargs):                   
        for value in kwargs.values():
            if not isinstance(value, self.input_type()):
                raise ValueError(f"Invalid input type. Expected {self.input_type()}.")        
        detections = super().run(**kwargs)                             
        names = []                                   
        for value in detections.values():            
            image: Image = value.get("image")
            classification_results: list = value.get("results", [])
            if not classification_results:
                continue            
            cls_result = classification_results.pop()                       
            coordinates = cls_result.boxes.xywhn
            names = [
                cls_result.names[cls.item()]
                for cls in cls_result.boxes.cls.int()
            ]                                                
            for index in range(len(coordinates)):
                x, y, w, h = coordinates[index]
                cropped = ImageCrop.build(
                    image.path(),
                    x_center=float(x),
                    y_center=float(y),
                    width_relative=float(w),
                    height_relative=float(h),
                )                     
                if self.output.get(names[index], None) is None:
                    self.output[names[index]] = [cropped]
                else:
                    self.output[names[index]].append(cropped)                     
            for key in self.output.keys():
                if self.output[key] and len(self.output[key]) == 1:
                    self.output[key] = self.output[key][0] 
    
    def save(self, output_path: Path, **kwargs) -> tuple[Path, dict, list[Path]]:
        
        keys_to_remove = kwargs.get("keys_to_remove", [])
        file_name = self.output.pop("file", "")
        for key in keys_to_remove:
            if key in self.output:
                self.output.pop(key)        
        result_dict = dict()
        file_name = ""
        image_path = ""
        for key, value in self.output.items():
            if isinstance(value, Image | ImageCrop):
                value = value.metadata()
            if isinstance(value, list):
                if all(isinstance(v, ImageCrop) for v in value):
                    image_path = value[0].path()
                    value = ImageCrop.metadata_list(value)                                        
                elif len(value) == 1 and isinstance(value[0], Image):                        
                    image_path = value[0].path()
                    value = value[0].metadata()
            if not value:
                continue
            result_dict.update({key: value})            
            if not file_name and not isinstance(value, str) and value.get("path"):
                file_name = f"{value.get("path")}.{kwargs.pop("format", "")}"
        if not file_name:
            file_name = f"output.{kwargs.pop("format", "")}"
        result_path = output_path / file_name.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")          
        return result_path, result_dict, [Path(image_path)] if image_path else []  
