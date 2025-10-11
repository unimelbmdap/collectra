"""Base machine learning task class for Collectra workflows.

This module defines the base machine learning task interface.
It inherits from the base Task class and implements the core task execution lifecycle.

Classes:
    MachineLearningTask: Base class for all machine learning tasks
"""

from typing import Generic

from collectra.types.base import T

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

    def __init__(self, name: str, model: T = None) -> None:
        super().__init__(name)
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
