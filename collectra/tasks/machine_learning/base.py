"""Base machine learning task class for Collectra workflows.

This module defines the base machine learning task interface.
It inherits from the base Task class and implements the core task execution lifecycle.

Classes:
    MachineLearningTask: Base class for all machine learning tasks
"""

from datetime import datetime
from pathlib import Path
from typing import Generic

from collectra.commons import T
from collectra.cli import command

from ..base import Task


class MachineLearningTask(Task, Generic[T]):
    """Base class for machine learning tasks in Collectra workflows.

    Provides common functionality for ML tasks including model loading,
    training, evaluation, and inference operations. Serves as the foundation
    for specialized ML task implementations.

    Attributes:
        VALID_MODEL: Class reference to the valid model type for this task.
    """

    model: T

    def __init__(self, name: str, model: T = None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.model = model

    def get_model(self) -> T:
        """Get the machine learning model associated with the task.

        Returns:
            T: The machine learning model.
        """
        return self.model

    def _load(self):
        """Load a machine learning model for this task.

        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement the load method.")

    def train(self, **kwargs) -> T:
        """Train the machine learning model.

        This method must be implemented by all subclasses.

        """
        raise NotImplementedError("Subclasses must implement the train method.")

    @command(name="train")
    def cli_train(
        self,
        inputs: list[str],
        keep_log: bool = True,
        validation: str = "",
        exclude: str = "",
    ) -> None:
        """Train this task using the supplied Collectra data."""
        if self.pipeline is None:
            raise RuntimeError(f"Task {self.name!r} is not attached to a pipeline")
        log = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pipeline.train(
            self.name,
            input=inputs,
            project=f"{self.pipeline.path.name}-{self.name}",
            log=log,
            base_folder=Path.cwd(),
            validation=validation,
            exclude=exclude,
        )
        self.pipeline.save(self.name)
        if not keep_log:
            import shutil

            shutil.rmtree(log, ignore_errors=True)

    def eval(self, **kwargs) -> None:
        """Evaluate the machine learning model performance.

        This method must be implemented by all subclasses.

        """
        raise NotImplementedError("Subclasses must implement the eval method.")

    def cluster(self, **kwargs) -> None:
        """Perform clustering analysis on the dataset.

        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement the cluster method.")

    def set_model(self, model: T) -> None:
        """Set the model associated with the task.

        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement the set_model method.")
