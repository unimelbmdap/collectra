from pathlib import Path
from dataclasses import dataclass, field
import tempfile, zipfile, pytz, shutil, yaml, os
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from .models import Model, YOLOModel
from datetime import datetime
from collectra.images import Image, ImageCrop

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

    def __call__(self, **kwargs):
        self.check_kwargs(**kwargs)
        return self.run(**kwargs)

    def check_kwargs(self, **kwargs) -> None:
        assert len(kwargs) == len(self.input), f"Number of arguments to {self} incorrect. Expected {len(self.input)} and received {len(len(kwargs))}"
        for key,value in kwargs.items():
            # TODO Check
            pass


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
        return self.load(self.config.get("model")).detect(self.config)  # type: ignore

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

    def run(self):
        results = super().run()
        names = []        
        for result in results:
            image_file = Path(result.get("image"))  # type: ignore
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
            path = Path(f"output/{image_file.stem}.{self.config['file_format']}")
            path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists            
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
            cls_result.save_crop(save_dir=path)            
            if not self.config.get("as_dir", False):
                shutil.make_archive(
                    str(path), "zip", path
                )  # Create a zip archive of the results
                shutil.rmtree(path)  # Remove the directory after zipping
                os.rename(f"{path}.zip", path.parent / f"{path.name}")    


@dataclass(kw_only=True)
class LLM(Task):
    model:str
    template:str|Path
    llm:LLM = field(init=False)
    temperature: float | None = None,
    max_tokens: int = None,
    variables: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        import llmloader
        
        self.llm = llmloader.load(self.model, temperature=self.temperature, max_tokens=self.max_tokens)

    def replace_in_template(self, prompt, key, value:str|Path):
        if Path(value).exists():
            value = Path(value)

        if isinstance(value, Path):
            if value.is_file():
                value = value.read_text()
            else:
                value = str(value)

        return prompt.replace(f"{{{key}}}", str(value))

    def run(self, **kwargs) -> str:
        prompt = str(self.template)
        
        # Replace inputs and config in template
        for key, value in kwargs.items() + self.variables.items():
            prompt = self.replace_in_template(prompt, key, value)
        
        result = self.llm.invoke(prompt)
        parser = StrOutputParser()
        return parser.invoke(result)
