import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from ..commons.files import CollectraFile
from .base import Converter


class YOLOConfig(BaseModel):
    """
    Configuration for YOLO dataset conversion.
    This class defines the structure of the YOLO configuration file, which includes:
    - Paths to the training and validation datasets,
    - The number of classes
    - The list of class names.
    """

    train: str = Field(..., description="Path to the training dataset")
    val: str = Field(..., description="Path to the validation dataset")
    nc: int = Field(..., description="Number of classes")
    names: list[str] = Field(..., description="List of class names")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "YOLOConfig":
        """Load YOLO configuration from a YAML file.

        Args:
            yaml_path (Path): Path to the YAML configuration file.

        Returns:
            YOLOConfig: The loaded YOLO configuration.
        """
        import yaml

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class YOLOConverter(Converter):
    """
    Converter for YOLO dataset format to Collectra format.
    This converter reads the YOLO configuration file, processes the training and validation datasets, and converts the annotations to Collectra format.
    The converted files are saved in the specified output directory.
    """

    input: Path
    output: Path
    root_label: str
    ext: str
    force: bool

    def __init__(
        self,
        input: str | Path,
        root_label: str,
        ext: str,
        output: str | Path,
        force: bool = False,
    ):
        """Initialize the YOLOConverter. Process the input configuration file or folder to extract necessary information for conversion.

        Args:
            input (str | Path): Path to the YOLO configuration file or folder.
            root_label (str): Root label for converted data.
            ext (str): File extension for converted collectra files.
            output (str | Path): Output folder for converted files.
            force (bool, optional): Whether to force overwrite existing files. Defaults to False.

        """
        self.input = Path(input)
        self.root_label = root_label
        self.ext = ext
        self.output = Path(output)
        self.force = force
        self._get_config()

    def _get_config(self) -> YOLOConfig:
        """Get the YOLO configuration from the input file or directory.

        Returns:
            YOLOConfig: The YOLO configuration.

        """
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
        return YOLOConfig.from_yaml(self.input)

    def _convert_file(
        self, file: Path, txt_file: Path, partition: str | None = None
    ) -> None:
        collectra_file = CollectraFile.from_file(
            file=file,
            label=self.root_label,
            ext=self.ext,
            output=self.output,
            force=self.force,
            partition=partition,
        )
        """ Read the YOLO annotation file and convert it to Collectra format. 
            The YOLO annotation file contains lines with the format:
            <class_id> <x_center> <y_center> <width_relative> <height_relative>
            Save the converted data in the Collectra file.

        Args:            
            file (Path): Path to the image file.
            txt_file (Path): Path to the YOLO annotation file.

        """
        with open(txt_file, "r") as f:
            crops = [line.strip().split(" ") for line in f if line.strip()]

        import uuid

        for crop in crops:
            class_id, x_center, y_center, width_relative, height_relative = crop
            class_id = int(class_id)
            class_name = self._get_config().names[class_id]
            if class_name not in collectra_file.data:
                collectra_file.data[class_name] = []
            if not isinstance(collectra_file.data[class_name], list):
                collectra_file.data[class_name] = [collectra_file.data[class_name]]
            collectra_file.data[class_name].append(
                {
                    "id": f"{class_name}-{uuid.uuid4().hex[:8]}",
                    "data": file.name,
                    "type": "collectra.ImageCrop",
                    "x_center": float(x_center),
                    "y_center": float(y_center),
                    "parents": self.root_label,
                    "width_relative": float(width_relative),
                    "height_relative": float(height_relative),
                }
            )

        collectra_file.save()

    def _process_files(self, input_of_files: Path, partition):
        """Process the files listed in the input file.
           The input file contains paths to image files, one per line.
           For each image file, find the corresponding YOLO annotation file and convert it to Collectra format.

        Args:
            input_of_files (Path): Path to the file containing paths to image files.

        Raises:
            ValueError: If any of the image files or corresponding annotation files do not exist.
            RuntimeError: If there is an error during file processing, the output directory will be cleaned

        """
        files: list[Path] = []
        text_files: list[Path] = []
        with open(input_of_files, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                file_path = (
                    Path(line.strip())
                    if str(line.strip()).startswith("/")
                    else input_of_files.parent / line.strip()
                )
                txt_path = Path(str(file_path).replace("images", "labels")).with_suffix(
                    ".txt"
                )
                if not file_path.exists() or not txt_path.exists():
                    raise ValueError(f"File {file_path} or {txt_path} does not exist.")
                files.append(file_path)
                text_files.append(txt_path)
        try:
            self.output.mkdir(parents=True, exist_ok=True)
            for file, txt_file in zip(files, text_files):
                self._convert_file(file, txt_file, partition=partition)
        except Exception as e:
            shutil.rmtree(self.output, ignore_errors=True)
            raise RuntimeError(f"Failed to process files: {e}")

    def convert(self) -> None:
        """Convert the YOLO dataset to Collectra format. Process both training and validation datasets as specified in the YOLO configuration."""
        config = self._get_config()
        self._process_files(
            self.input.parent / config.train, partition=str(Path(config.train).stem)
        )
        self._process_files(
            self.input.parent / config.val, partition=str(Path(config.val).stem)
        )


class YOLOClassifierCSV:
    """
    Converter for YOLO classifier dataset format to Collectra format.
    This converter extends the YOLOConverter to handle classification datasets in YOLO format.
    It processes the training and validation datasets, converts the annotations to Collectra format, and saves the converted files in the specified output directory.
    """

    def __init__(
        self,
        input: str | Path,
        root_label: str,
        ext: str,
        output: str | Path,
        force: bool = False,
    ):
        """Initialize the YOLOConverter. Process the input configuration file or folder to extract necessary information for conversion.

        Args:
            input (str | Path): Path to the YOLO configuration file or folder.
            root_label (str): Root label for converted data.
            ext (str): File extension for converted collectra files.
            output (str | Path): Output folder for converted files.
            force (bool, optional): Whether to force overwrite existing files. Defaults to False.

        """
        self.input = Path(input)
        self.root_label = root_label
        self.ext = ext
        self.output = Path(output)
        self.force = force

    def convert(self) -> None:
        import pandas as pd

        config_file = pd.read_csv(self.input)
        for file_data in config_file.itertuples():
            source_file = self.input.parent / str(file_data.path)
            label_data = str(file_data.tag)
            validation = "validation" if file_data.validation else "train"
            print(
                f"Processing file {source_file} with label {label_data} for partition {validation}"
            )
            self._convert_file(source_file, label_data, partition=validation)

    def _convert_file(
        self, file: Path, label_data: str, partition: str | None = None
    ) -> None:
        import uuid

        collectra_file = CollectraFile.from_file(
            file=file,
            label=self.root_label,
            ext=self.ext,
            output=self.output,
            force=self.force,
            partition=partition,
        )
        collectra_file.data[f"{self.root_label}_writing_type"] = {
            "id": f"{self.root_label}_writing_type-{uuid.uuid4().hex[:8]}",
            "type": "collectra.Text",
            "parents": collectra_file.data[self.root_label]["id"],
            "data": label_data,
        }
        collectra_file.save()
