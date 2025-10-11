"""Workflow management and configuration loading for Collectra pipelines.

This module provides the CollectraManager class which handles loading and
building Collectra workflows from various sources including directories
and zip files. It manages the complete workflow lifecycle from configuration
parsing to execution setup.

The manager supports:
    - Loading workflows from directories and zip files
    - Building workflow configurations from metadata
    - Task instantiation and dependency setup
    - Configuration validation and preprocessing

Classes:
    CollectraManager: Main manager class for workflow operations
"""

import os
from dataclasses import dataclass, field
from collectra.pipelines.base import Collectra
from collectra.pipelines.handlers import DirectoryHandler, ZipHandler
from collectra.tasks.managers import TaskManager
from collectra.utils import success_msg, processing_msg, from_dir, unzip
from pathlib import Path
from rich import print


@dataclass(kw_only=True)
class CollectraManager:
    """Manager class for Collectra workflow operations.

    Handles the complete lifecycle of Collectra workflows including loading
    from various sources, building from metadata, and saving to different formats.

    Attributes:
        pipeline (Collectra | None): The managed workflow pipeline instance.
        required_keys (list[str]): Required metadata keys for valid workflows.
    """

    pipeline: Collectra | None = None
    required_keys: list[str] = field(
        default_factory=lambda: ["name", "version", "description", "format"]
    )

    def build(self, metadata):
        """Build a Collectra workflow from metadata.

        Args:
            metadata: Dictionary containing workflow metadata and configuration.
        """
        self.pipeline = Collectra(**metadata)

    def load(self, pipeline: Path):
        """Load an existing Collectra workflow from a file or directory.

        Automatically detects whether the source is a directory or zip file
        and loads the workflow configuration accordingly.

        Args:
            pipeline (Path): Path to the workflow directory or zip file.

        Raises:
            ValueError: If the path is not a valid directory or file.
        """
        if pipeline.is_dir():
            self._load_dir(pipeline)
        elif pipeline.is_file():
            self._load_zip(pipeline)
        else:
            raise ValueError(f"Invalid collectra file. Please provide a valid path!")

    def get_pipeline(self) -> Collectra:
        """Get the managed workflow pipeline.

        Returns:
            Collectra: The loaded or built workflow pipeline.

        Raises:
            ValueError: If no pipeline has been initialized.
        """
        if not self.pipeline:
            raise ValueError(
                "Pipeline is not initialized. Please load a pipeline first."
            )
        return self.pipeline

    def _load_dir(self, path: Path):
        """Load an existing Collectra workflow from a directory.

        Reads workflow configuration from the specified directory and initializes
        the pipeline instance with the loaded data.

        Args:
            path (Path): The path to the directory containing the workflow configuration.
        """
        # Save CWD
        cwd = Path.cwd()
        if path.is_dir():
            os.chdir(path)

        data = from_dir(path)
        self._load_pipeline(data, path, as_dir=True)

        # Revert current working directory
        os.chdir(cwd)

    def _load_zip(self, path: Path):
        """Load an existing Collectra workflow from a zip file.

        Extracts and reads workflow configuration from the specified zip file
        and initializes the pipeline instance with the loaded data.

        Args:
            path (Path): The path to the zip file containing the workflow configuration.
        """
        data = unzip(path)
        self._load_pipeline(data, path, as_dir=False)

    def _load_pipeline(
        self, data: dict = dict(), path: Path | None = None, as_dir: bool = False
    ):
        """Parse workflow configuration data and create a Collectra instance.

        Validates required metadata keys, sets up output directory, and initializes
        the pipeline with the provided configuration data.

        Args:
            data (dict, optional): The workflow configuration dictionary. Defaults to dict().
            path (Path | None, optional): Path to the workflow source. Defaults to None.
            as_dir (bool, optional): Whether the workflow source is a directory.
                                   Defaults to False.

        Raises:
            ValueError: If required pipeline metadata keys are missing.
        """
        metadata = data.pop("collectra_pipeline_metadata", dict())
        if not set(metadata.keys()).issubset(set(self.required_keys)):
            raise ValueError(
                f"Invalid pipeline file. Must have the keys: {', '.join(self.required_keys)} in metadata."
            )
        metadata["out_dir"] = str(path) if path else str(Path.cwd() / metadata["name"])
        metadata["as_dir"] = as_dir
        self.build(metadata)
        self._setup(data)

    def _setup(self, config: dict) -> None:
        """Load configuration from a dictionary and populate the workflow with tasks.

        Creates task instances from the configuration data, adds them to the pipeline,
        and establishes connections between tasks based on their dependencies.

        Args:
            config (dict): A dictionary containing the workflow task configurations.

        Raises:
            ValueError: If the pipeline is not initialized.
        """
        if not self.pipeline:
            raise ValueError(
                "Pipeline is not initialized. Please load a pipeline first."
            )

        mapping = dict()
        for name, data in config.items():
            data = self._modify_config_data(name, data)
            task_instance = TaskManager.build(data)
            self.pipeline.add_task(task_instance, mapping)
            print(success_msg(f"Loaded {task_instance.name}"))

        self.pipeline.connect(mapping)

        if len(self.pipeline.tasks) == 0:
            print(processing_msg("No tasks found in the workflow configuration."))

    def _modify_config_data(self, name: str, data: dict) -> dict:
        """Modify task configuration data for proper pipeline setup.

        Processes task configuration by setting the task name, adjusting model paths
        to absolute paths, and ensuring input/output fields are in list format.

        Args:
            name (str): The name of the task.
            data (dict): The task configuration dictionary.

        Returns:
            dict: The modified configuration dictionary.
        """
        data["name"] = name
        if data.get("model", None) is not None:
            data["model"] = f"{self.pipeline.out_dir}/{data.get('model')}"  # type: ignore
            data["old_model"] = data["model"]
        if data.get("input", None) is not None:
            data["input"] = (
                [data["input"]] if isinstance(data["input"], str) else data["input"]
            )
        if data.get("output", None) is not None:
            data["output"] = (
                [data["output"]] if isinstance(data["output"], str) else data["output"]
            )
        return data

    def save(self) -> None:
        """Save the workflow pipeline using the appropriate handler.

        Determines whether to use DirectoryHandler or ZipHandler based on the
        pipeline's format setting and saves the workflow configuration and artifacts.

        Raises:
            ValueError: If the pipeline is not initialized.
        """
        if not self.pipeline:
            raise ValueError(
                "Pipeline is not initialized. Please load a pipeline first."
            )
        if self.pipeline.as_dir:
            DirectoryHandler(pipeline=self.pipeline).save()
        else:
            ZipHandler(pipeline=self.pipeline).save()
