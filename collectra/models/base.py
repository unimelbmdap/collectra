from abc import ABC, abstractmethod
from pathlib import Path
import os

class Model(ABC):
    """
    Abstract base class for all machine learning models.

    This class defines the common interface and behavior for all machine learning
    models in the Collectra system. It provides a standardized way to train,
    validate, detect, and manage machine learning models.

    Attributes:
        DEFAULT_CONFIG (Dict): Default configuration parameters for the model
    """

    def __init__(self, path: str | Path, config: dict = {}):
        """
        Initialize a Model instance.

        Args:
            model (str | Path): Name or path to the model file
            config (dict, optional): Configuration parameters for the model.
                                   Defaults to empty dict.
        """
        self.path: Path = Path(path)
        self.config: dict = config

    def __str__(self) -> str:
        """
        Return string representation of the model.

        Returns:
            str: String representation showing the model name
        """
        return f"{self.path.name}"

    def get_path(self) -> Path:
        return self.path

    def delete(self) -> None:
        if self.path and self.path.exists():
            os.remove(self.path)

    @abstractmethod
    def train(self, config: dict = {}) -> Path | None:
        """
        Train the machine learning model with the given configuration.

        Args:
            config (dict, optional): Training configuration parameters.
                                   Defaults to empty dict.

        Returns:
            Path | None: Path to the trained model file, or None if training failed
        """
        pass

    @abstractmethod
    def val(self, config: dict = {}) -> None:
        """
        Validate the trained model to assess its performance.

        This method should implement model validation logic specific to
        the model type and return or display validation metrics.
        """
        pass

    @abstractmethod
    def detect(self, config: dict = {}) -> list[dict]:
        """
        Run inference/detection on the provided data.

        Args:
            data (Path): Path to the data to process
        """
        pass

    @abstractmethod
    def preprocess(self, data: Path, file_format: str) -> None:
        """
        Preprocess data for training or inference.

        Args:
            data (Path): Path to the data to preprocess
            file_format (str): Format of the input files
        """
        pass
