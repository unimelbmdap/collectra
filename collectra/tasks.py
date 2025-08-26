from pathlib import Path
from .models import Model, YOLOModel
from dataclasses import dataclass, field
import tempfile, zipfile


@dataclass(kw_only=True)
class Task:
    name: str
    input: list[str] = field(default_factory=list)
    output: list[str] = field(default_factory=list)
    config: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, name: str, **kwargs) -> "Task":
        input = kwargs.pop("input", [])
        output = kwargs.pop("output", [])
        return cls(name=name, input=input, output=output, config=kwargs)

    def __post_init__(self):
        if isinstance(self.input, str):
            self.input = [self.input]
        if isinstance(self.output, str):
            self.output = [self.output]                     

    def metadata(self) -> dict:
        """
        Get metadata of the task.
        :return: A dictionary containing task metadata.
        """
        metadata = {
            "name": self.name,
            "model": self.config.get("model", ""),
        }
        if self.input:
            metadata["input"] = self.input if len(self.input) > 1 else self.input[0]  # type: ignore
        if self.output:
            metadata["output"] = self.output if len(self.output) > 1 else self.output[0]  # type: ignore
        return metadata

    def __str__(self) -> str:
        return f"{self.name} of {self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"{self.name} of {self.__class__.__name__}"

    def set_config(self, config: dict) -> None:
        """
        Set the configuration for the task.
        :param config: A dictionary containing configuration parameters.
        """
        if not isinstance(config, dict):
            raise ValueError("Invalid config type.")
        self.config.update(config)

    def run(self) -> list[dict] | None:
        """
        Run the task.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")    

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

    def run(self) -> list[dict]:
        if not self.config.get("model", None):
            raise ValueError("Model must be set before running the task.")
        return self.load(self.config.get("model")).detect(self.config) # type: ignore

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
                with zipfile.ZipFile(model, 'r') as zip_ref:
                    zip_ref.extract(member=model.name, path=tmpdirname)
                    model_path = Path(tmpdirname) / model.name
                    return self.load(model_path).train(self.config) # type: ignore

    def eval(self) -> None:
        if not self.config.get("model", None):
            raise ValueError("Model must be set before validating the task.")
        self.load(self.config.get("model")).val(self.config) # type: ignore

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



