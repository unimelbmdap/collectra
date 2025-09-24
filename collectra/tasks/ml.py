from pathlib import Path
from collectra.images.base import Image, ImageCrop
from collectra.models.base import Model
from collectra.models.yolo import YOLOModel
from collectra.tasks.base import Task
from dataclasses import dataclass
from datetime import datetime
from types import UnionType
import tempfile, zipfile, pytz, shutil, yaml, os

@dataclass(kw_only=True)
class MachineLearningTask(Task):

    VALID_MODEL = None

    def __post_init__(self):
        super().__post_init__()

    def load(self, model: str | Path) -> Model:
        if not self.VALID_MODEL or not model:
            raise Exception(
                "No valid model type defined for this task or model path is empty."
            )        
        return self.VALID_MODEL(model)

    def run(self, **kwargs) -> list[dict]:
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
        return Image | Path
    
    def output_type(self) -> type:
        return ImageCrop

    def run(self, **kwargs):        
        results = super().run(**kwargs)        
        names = []        
        for result in results:            
            image_file = Path(result.get("image"))  # type: ignore
            path = Path(f"{image_file.stem}.{self.config['format']}")
            path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists            
            output_yaml = {
                "collectra_results_metadata": {
                    "timestamp": datetime.now(pytz.utc).isoformat(),
                    "validation": False,
                },
                "specimen_sheet": {
                    "type": f"{Image.__module__}.{Image.__name__}",
                    "path": image_file.name,
                },
            }
            classification_results = result.get("results", [])
            if not classification_results:
                continue
            for cls_result in classification_results:
                coordinates = cls_result.boxes.xywhn
                names = [
                    cls_result.names[cls.item()]
                    for cls in cls_result.boxes.cls.int()
                ]
                for index in range(len(coordinates)):
                    x, y, w, h = coordinates[index]
                    output_yaml[names[index]] = {
                        "type": f"{ImageCrop.__module__}.{ImageCrop.__name__}",
                        "path": image_file.name,
                        "x_center": float(x),
                        "y_center": float(y),
                        "width_relative": float(w),
                        "height_relative": float(h),
                    }
                cls_result.save_crop(save_dir=path)            
            shutil.copy(image_file, path / image_file.name)
            with open(path / "results.yaml", "w") as f:
                for key in output_yaml:
                    f.write(
                        yaml.dump(
                            {key: output_yaml[key]},
                            default_flow_style=False,
                            sort_keys=False,
                        )
                    )
                    f.write("\n")                     
            if not self.config.get("as_dir", False):
                shutil.make_archive(
                    str(path), "zip", path
                )  # Create a zip archive of the results
                shutil.rmtree(path)  # Remove the directory after zipping
                os.rename(f"{path}.zip", path.parent / f"{path.name}")    

