from pathlib import Path

from pydantic import BaseModel, Field

from .base import Converter


class YOLOConfig(BaseModel):
    train: str = Field(..., description="Path to the training dataset")
    val: str = Field(..., description="Path to the validation dataset")
    nc: int = Field(..., description="Number of classes")
    names: list[str] = Field(..., description="List of class names")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "YOLOConfig":
        import yaml

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class YOLOConverter(Converter):

    input: Path

    def __init__(self, input: str | Path):
        self.input = Path(input)
        self._get_config()

    def _process_config(self) -> YOLOConfig:
        config = YOLOConfig.from_yaml(self.input)
        return config

    def _get_config(self) -> YOLOConfig:
        if not (self.input.is_file() or self.input.is_dir()):
            raise ValueError(
                f"Input path {self.input} is neither a file nor a directory."
            )
        if self.input.is_file():
            if self.input.suffix != ".yaml":
                raise ValueError(
                    f"Expected a YAML config file, but got {self.input.suffix}"
                )
        if self.input.is_dir():
            config_files = list(self.input.glob("*.yaml"))
            if not config_files:
                raise ValueError(
                    f"No YAML config files found in directory {self.input}"
                )
            if len(config_files) > 1:
                raise ValueError(
                    f"Multiple YAML config files found in directory {self.input}. Please ensure there is only one config file."
                )
            self.input = config_files[0]
        return self._process_config()

    def convert(self):
        config = self._get_config()
        print(config)
