from dataclasses import dataclass, field
from collectra.pipelines.base import Collectra
from collectra.pipelines.handlers import DirectoryHandler, ZipHandler
from collectra.tasks.managers import TaskManager
from collectra.utils import success_msg, processing_msg
from pathlib import Path
from rich import print
import yaml, zipfile

@dataclass(kw_only=True)
class CollectraManager:

    pipeline: Collectra | None = None
    required_keys: list[str] = field(default_factory=lambda: ["name", "version", "description", "file_format"])

    def build(self, metadata):
        self.pipeline = Collectra(**metadata)                         

    def load(self, pipeline: Path):
        """
        Load an existing Collectra workflow from a YAML file.

        This method reads the workflow configuration from the specified path.
        """
        if pipeline.is_dir():
            self._load_dir(pipeline)
        elif pipeline.is_file():
            self._load_zip(pipeline)
        else:
            raise ValueError(f"Invalid collectra file. Please provide a valid path!")

    def get_pipeline(self) -> Collectra:
        if not self.pipeline:
            raise ValueError("Pipeline is not initialized. Please load a pipeline first.")
        return self.pipeline
    
    def _load_dir(self, path: Path):
        """
        Load an existing Collectra workflow from a directory.
        This method reads the workflow configuration from the specified directory.
        :param path: The path to the directory containing the workflow configuration.
        :return: An instance of Collectra with the loaded configuration.
        """
        pipeline_file = path / "pipeline.yaml"        
        with open(pipeline_file, "r") as f:
            data = yaml.safe_load(f)
            if not data:
                raise ValueError(f"Pipeline file is empty or invalid: {pipeline_file}")
            self._load_pipeline(data, path, as_dir=True)

    def _load_zip(self, path: Path):
        """
        Load an existing Collectra workflow from a zip file.
        This method reads the workflow configuration from the specified zip file.
        :param path: The path to the zip file containing the workflow configuration.
        :return: An instance of Collectra with the loaded configuration.
        """
        pipeline_file = "pipeline.yaml"
        with zipfile.ZipFile(path, "r") as zipf:
            data = yaml.safe_load(zipf.read(pipeline_file))            
            self._load_pipeline(data, path, as_dir=False)
    
    def _load_pipeline(
        self,
        data: dict = dict(), 
        path: Path | None = None, 
        as_dir: bool = False
    ):
        """
        Parse the workflow configuration data and create a Collectra instance.
        :param data: the workflow configuration.
        :param as_dir: If True, the workflow is a directory instead of a file. A flag is saved in the Collectra instance.

        """
        metadata = data.pop("collectra_pipeline_metadata", dict())   
        if not set(metadata.keys()).issubset(set(self.required_keys)):        
            raise ValueError(f"Invalid pipeline file. Must have the keys: {", ".join(self.required_keys)} in metadata.")                                 
        metadata["out_dir"] = str(path) if path else str(Path.cwd() / metadata["name"])
        metadata["as_dir"] = as_dir 
        self.build(metadata)        
        self._setup(data)        

    def _setup(self, config: dict) -> None:
        """
        Load the configuration from a dictionary.
        This method populates the workflow with tasks based on the provided configuration.
        :param config: A dictionary containing the workflow configuration.
        """
        if not self.pipeline:
            raise ValueError("Pipeline is not initialized. Please load a pipeline first.")            

        for name, data in config.items():               
            data["name"] = name
            task_instance = TaskManager.build(data)
            self.pipeline.tasks.append(task_instance)
            print(success_msg(f"Loaded {task_instance.name}"))

        if len(self.pipeline.tasks) == 0:
            print(processing_msg("No tasks found in the workflow configuration."))

    def save(self) -> None:
        if not self.pipeline:
            raise ValueError("Pipeline is not initialized. Please load a pipeline first.")            
        if self.pipeline.as_dir:
            DirectoryHandler(pipeline=self.pipeline).save() 
        else:
            ZipHandler(pipeline=self.pipeline).save()
