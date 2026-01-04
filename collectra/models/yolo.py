"""YOLO object detection model implementation for Collectra workflows.

This module provides a complete implementation of the Model abstract class
for YOLO-based object detection models using the Ultralytics framework.
It handles the full machine learning pipeline including training, validation,
inference, and data preprocessing specific to YOLO models.

The YOLOModel class supports:
    - Training with custom datasets and configurations
    - Model validation and performance evaluation
    - Object detection inference with bounding box results
    - Data preprocessing and YOLO format conversion
    - Thread-safe detection for concurrent processing
    - Automatic GPU acceleration on macOS (MPS) and other platforms

Classes:
    YOLOModel: Complete YOLO model implementation with training and inference
"""

import importlib
import os
import platform
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml
from rich import print
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import ThreadingLocked

from collectra.models.base import Model
from collectra.types.images import Image, ImageCrop
from collectra.utils import error_msg, get_all_files, processing_msg, success_msg


class YOLOModel(Model):
    """Concrete implementation of the Model abstract class for YOLO-based object detection.

    This class provides a complete implementation for training, validating, and running
    inference with YOLO models using the Ultralytics framework. It handles the full
    machine learning pipeline including data preprocessing, model training, validation,
    and object detection inference.

    Features:
        - Training with custom datasets and hyperparameter configuration
        - Model validation and performance evaluation with metrics
        - Thread-safe object detection inference
        - Automatic data preprocessing and YOLO format conversion
        - GPU acceleration support (MPS on macOS, CUDA on other platforms)
        - Clustering analysis for dataset exploration

    Attributes:
        DEFAULT_CONFIG (dict): Default configuration parameters for YOLO training
                             including epochs (50), image size (640), verbosity (True),
                             and initial learning rate (0.01).
        model (YOLO): The underlying YOLO model instance from Ultralytics.
        yolo_config_path (str): Path to the YOLO dataset configuration file.
        dir (Path): Directory path for training outputs and logs.
    """

    DEFAULT_CONFIG: dict = {
        "epochs": 50,  # Number of training epochs
        "imgsz": 640,  # Input image size for training
        "verbose": True,  # Enable verbose output during training
        "lr0": 0.01,  # Initial learning rate
    }

    def __init__(self, path: str | Path, config: dict = {}):
        """
        Initialize a YOLOModel instance.

        Args:
            name (str | Path, optional): Path to the YOLO model file.
            config (dict, optional): Configuration parameters for the model.
        """
        super().__init__(path, config)
        print(processing_msg(f"Loading YOLO model with path {path}"))
        self.model = YOLO(path, verbose=True)
        self.yolo_config_path: str = "config.yml"

    def _setup_environment(self, config: dict) -> None:
        """
        Setup the training environment and create necessary directories.

        This method creates a unique timestamped directory for the training
        session to avoid conflicts with concurrent training runs.

        Args:
            config (dict): Configuration dictionary containing directory settings
        """
        # Create unique directory with timestamp to avoid conflicts
        self.dir = Path(config.get("output_log", "logs"))
        self.dir.mkdir(parents=True, exist_ok=True)
        print(success_msg(f"Successfully setup training environment at {self.dir}"))

    def train(self, config: dict = {}) -> Path:
        """Train the YOLO model with the given configuration.

        Executes the complete training pipeline including environment setup,
        data preparation, model training, validation, and result saving.

        Args:
            config (dict, optional): Configuration dictionary containing training
                                   parameters such as epochs, learning rate, data paths,
                                   and output settings. Defaults to {}.

        Returns:
            Path: Path to the trained model weights file (best.pt).
        """
        merged_config = {**self.config, **config}
        print(
            processing_msg(
                f"[bold green]Training object detection model[/bold green]: {self.path}"
            )
        )
        self._setup_environment(merged_config)
        self._prepare_data(merged_config)
        train_results = self._execute_training(merged_config)
        metrics = self._validate_model()
        self._save_results(merged_config, train_results, metrics)
        return self._get_model_path()

    def _prepare_data(self, config: dict) -> None:
        """
        Prepare and preprocess training data.

        Extracts input data path and file format from configuration,
        then calls the preprocessing method to convert data to YOLO format.

        Args:
            config (dict): Configuration dictionary containing input settings
        """
        data = config.get("input_files", None)
        inputs = config.get("inputs", [])
        classes = config.get("outputs", [])
        if isinstance(classes, dict):
            classes = list(classes.keys())
        if isinstance(inputs, dict):
            inputs = list(inputs.keys())
        if not data:
            raise ValueError(
                error_msg("Input data path is required for training. Found none")
            )
        format = config.get("format", None)
        if not format:
            raise ValueError(
                error_msg("File format is required for training. Found none")
            )
        format = format.replace(".", "")
        self.preprocess(data, inputs, classes, format=format)

    def _prepare_assets(
        self,
        file: Path,
        is_file: bool = True,
        inputs: list[str] = [],
    ) -> dict:
        """Prepare assets for YOLO training by extracting metadata and copying image files.

        This method reads the results.yaml file from the specified path and extracts
        image information for training data preparation. It also copies the image
        files to the training directory.

        Args:
            file (Path): Path to the file or directory containing training assets.
            is_file (bool, optional): Whether the path points to a file. Defaults to True.
            inputs (list[str], optional): List of input field names. Defaults to [].

        Returns:
            dict: Dictionary containing the loaded asset data from results.yaml,
                  empty dict if results.yaml is not found or invalid.
        """
        results_yaml = self.dir / "results.yaml" if is_file else file / "results.yaml"
        if not results_yaml.exists():
            print(
                error_msg(f"Results YAML file not found in {file} - Skipping {file}...")
            )
            return dict()
        with open(results_yaml, "r") as f:
            data = yaml.safe_load(f)
            image = data.get(inputs[0], None).get("path", None)
            if not image:
                print(error_msg(f"Invalid input for {file}. Skipping this..."))
                return dict()
            shutil.copy(file / image, self.dir / image)
        return data

    def _prepare_yolo_config(
        self,
        train_files: list[str],
        val_files: list[str],
        classes: list[str],
        validation: bool = False,
    ) -> None:
        """Prepare YOLO dataset configuration file and train/validation file lists.

        Creates the config.yml file required by YOLO training with dataset paths,
        number of classes, and class names. Also generates train.txt and val.txt
        files containing the paths to training and validation images.

        Args:
            train_files (list[str]): List of paths to training image files.
            val_files (list[str]): List of paths to validation image files.
            classes (list[str]): List of class names for object detection.
            validation (bool, optional): Whether this is for validation only.
                                       Defaults to False.
        """
        # Create YOLO dataset configuration
        yolo_config = ""
        if not validation:
            yolo_config = (
                f"train: train.txt\nval: val.txt\nnc: {len(classes)}\nnames: {classes}"
            )
            Path(f"{self.dir}/train.txt").write_text("\n".join(train_files))
        else:
            val_files.extend(train_files)
            yolo_config = f"val: val.txt\nnc: {len(classes)}\nnames: {classes}"
        Path(f"{self.dir}/train.txt").write_text("\n".join(train_files))
        Path(f"{self.dir}/val.txt").write_text("\n".join(val_files))
        Path(f"{self.dir}/config.yml").write_text(yolo_config)

    def _generate_annotation_str(self, item: dict) -> str:
        """Generate YOLO format annotation string from bounding box data.

        Converts bounding box information to the YOLO annotation format:
        "class_id x_center y_center width_relative height_relative"

        Args:
            item (dict): Dictionary containing bounding box data with keys:
                        'class_id', 'x_center', 'y_center', 'width_relative', 'height_relative'.

        Returns:
            str: YOLO format annotation string with newline character.
        """
        return f"{item['class_id']} {item['x_center']} {item['y_center']} {item['width_relative']} {item['height_relative']}\n"

    def _prepare_annotations(
        self, data_files: list[dict], classes: list[str]
    ) -> tuple[list[str], list[str]]:
        """Prepare YOLO annotation files from processed data files.

        Processes data files to generate YOLO format annotation files (.txt)
        for each image. Separates files into training and validation sets
        based on metadata flags.

        Args:
            data_files (list[dict]): List of dictionaries containing processed data
                                   with image and annotation information.
            classes (list[str]): List of class names for mapping class indices.

        Returns:
            tuple[list[str], list[str]]: Tuple containing (train_files, val_files)
                                       lists of image paths for training and validation.
        """
        train_files = []
        val_files = []

        for data in data_files:
            for_validation = False
            image = None
            annotation_str = ""
            for key, value in data.items():
                image_class = None
                if key == "collectra_results_metadata":
                    for_validation = value.get("validation", False)
                if value.get("type", None):
                    module_name, class_name = value["type"].rsplit(".", 1)
                    image_class = getattr(
                        importlib.import_module(module_name), class_name
                    )
                if image_class == Image:
                    image = value.get("path", None)
                if image_class == ImageCrop:
                    if "items" not in value:
                        value["class_id"] = classes.index(key)
                        annotation_str += self._generate_annotation_str(value)
                        continue
                    items = value.get("items", [])
                    for item in items:
                        item["class_id"] = classes.index(key)
                        annotation_str += self._generate_annotation_str(item)
            if not image:
                print(error_msg(f"Image not found in {data}. Skipping this..."))
                continue
            image = Path(image)
            annotation_file = self.dir / f"{image.stem}.txt"
            annotation_file.write_text(annotation_str)
            if for_validation:
                val_files.append(f"./{image}")
            else:
                train_files.append(f"./{image}")
        return train_files, val_files

    def preprocess(
        self,
        data_paths: list[str],
        inputs: list[str],
        classes: list[str],
        format: str,
        validation: bool = False,
    ) -> None:
        """Preprocess data files for YOLO training format.

        Processes input data files (zip archives or directories) and converts
        them to YOLO training format including image extraction, annotation
        generation, and dataset configuration file creation.

        Args:
            data_paths (list[str]): List of paths to data files or directories.
            inputs (list[str]): List of input field names to process.
            classes (list[str]): List of class names for object detection.
            format (str): File format extension for filtering files.
            validation (bool, optional): Whether to prepare for validation only.
                                       Defaults to False.
        """
        files: list[Path] = get_all_files(data_paths, format)
        data_files = []
        for file in tqdm(files, desc="Processing files for YOLO training"):
            data: dict | None = None
            if file.is_file():
                with tempfile.TemporaryDirectory() as tmpdirname:
                    with zipfile.ZipFile(file, "r") as zip_ref:
                        zip_ref.extractall(path=tmpdirname)
                        data = self._prepare_assets(tmpdirname / file, inputs=inputs)
            if file.is_dir():
                data = self._prepare_assets(file, is_file=False, inputs=inputs)
            if data:
                data_files.append(data)
        train_files, val_files = self._prepare_annotations(data_files, classes=classes)
        self._prepare_yolo_config(
            train_files, val_files, classes, validation=validation
        )

    def val(self, config: dict = {}) -> None:
        """Validate the YOLO model performance on test data.

        Performs model validation using the configured test dataset. Can run in
        test mode where no actual validation is performed for testing purposes.
        Results are saved to the output directory.

        Args:
            config (dict, optional): Configuration dictionary containing validation
                                   parameters and settings. Defaults to {}.
        """
        merged_config = {**self.config, **config}
        self._setup_environment(merged_config)
        print(f"[bold green]Training object detection model[/bold green]: {self.path}")
        self._prepare_data(merged_config)
        if merged_config.get("test", False):
            print(
                "[bold yellow]Test mode enabled[/bold yellow]: Validation will not be performed."
            )
            val_results = "Test mode: No training performed."
            metrics = "Test mode: No metrics available."
            self._save_results(merged_config, val_results, metrics, eval=True)
            return None
        metrics = val_results = self._execute_validation(merged_config)
        self._save_results(merged_config, val_results, metrics, eval=True)

    @ThreadingLocked()
    def thread_safe_detect(self, path: Path) -> list[Results]:
        """Thread-safe object detection inference.

        Performs object detection inference in a thread-safe manner using
        the ThreadingLocked decorator to prevent concurrent access issues.

        Args:
            path (Path): Path to the image file for detection.

        Returns:
            list[Results]: List of YOLO Results objects containing detection
                          information including bounding boxes and confidence scores.
        """
        return self.model.predict(path)

    def detect(self, **kwargs) -> dict:
        """Run object detection inference on provided image data.

        This method performs object detection on input images using the trained
        YOLO model. Results include bounding boxes, confidence scores, and class
        predictions for each detected object.

        Args:
            **kwargs: Variable keyword arguments where values can be Image objects
                     or lists of Image objects for detection. Keys become the
                     identifiers for the results.

        Returns:
            dict: Dictionary mapping input keys to detection results, where each
                  result contains the original image(s) and YOLO Results objects.

        Raises:
            ValueError: If any input value is not an Image object or list of Image objects.
        """
        print(f"[bold green]Running object detection[/bold green]: {self.path}")
        detections: dict = dict()
        for key, value in kwargs.items():
            if isinstance(value, Image):
                # Single image case
                detections[key] = {
                    "image": value,
                    "results": self.thread_safe_detect(value.data),
                }
            elif isinstance(value, list) and all(
                isinstance(img, Image) for img in value
            ):
                # Multiple images case
                detections[key] = []
                for img in value:
                    detections[key].append(
                        {"image": img, "results": self.thread_safe_detect(img.path)}
                    )
            else:
                raise ValueError(
                    f"Invalid input type for {key}. Expected Image or list of Images."
                )
        return detections

    def cluster(self, config) -> None:
        """Perform clustering analysis on the dataset.

        Sets up the environment and prepares data for clustering analysis.
        This method is typically used for data exploration and understanding
        the distribution of training data.

        Args:
            config (dict): Configuration dictionary containing clustering parameters
                          and data preparation settings.
        """
        merged_config = {**self.config, **config}
        self._setup_environment(merged_config)
        self._prepare_data(merged_config)

    def _execute_training(self, config: dict):
        """
        Execute the actual YOLO model training process.

        Configures training parameters and runs the YOLO training algorithm.
        Automatically detects macOS systems and uses Metal Performance Shaders (MPS)
        for GPU acceleration when available.

        Args:
            config (dict): Configuration dictionary with training parameters

        Returns:
            Training results object from YOLO training
        """
        # Prepare training parameters
        params = config.get("params", dict())
        params["data"] = Path(f"{self.dir}/{self.yolo_config_path}")
        params["project"] = self.dir
        if platform.system() == "Darwin":
            params["device"] = "mps"
        if params.get("epochs", None) is None:
            params["epochs"] = 1
        results = self.model.train(**params)
        return results

    def _execute_validation(self, config: dict):
        """Execute the YOLO model validation process.

        Runs validation on the prepared dataset using YOLO's built-in validation
        functionality. Configures validation parameters including data path,
        output directory, and device settings.

        Args:
            config (dict): Configuration dictionary containing validation parameters
                          such as verbosity and output settings.

        Returns:
            Validation results object containing performance metrics from YOLO validation.
        """
        print(f"[bold green]Validating YOLO model[/bold green]: {self.path}")
        validation_params = {
            "data": Path(f"{self.dir}/{self.yolo_config_path}"),
            "project": self.dir,
            "verbose": config.get("verbose", self.DEFAULT_CONFIG["verbose"]),
            "project": self.dir,
        }
        if platform.system() == "Darwin":
            validation_params["device"] = "mps"

        return self.model.val(**validation_params)

    def _get_model_path(self) -> Path:
        """
        Get the path to the best trained model weights.

        YOLO saves the best performing model during training in the
        'train/weights/best.pt' subdirectory of the project folder.

        Returns:
            Path: Path to the best model weights file
        """
        return Path(self.dir) / "train" / "weights" / "best.pt"

    def _validate_model(self):
        """
        Validate the trained YOLO model and return performance metrics.

        Runs validation on the test dataset to assess model performance
        and returns metrics such as mAP, precision, and recall.

        Returns:
            Validation metrics object from YOLO validation
        """
        return self.model.val()

    def _save_results(self, config: dict, train_results, metrics, eval=False) -> None:
        """
        Save training results, logs, and model artifacts to output directory.

        This method organizes and saves all training outputs including:
        - Training logs with results and metrics
        - Model weights and training artifacts
        - Validation results and plots

        Args:
            config (dict): Configuration dictionary containing output settings
            train_results: Training results object from YOLO training
            metrics: Validation metrics from model evaluation
        """
        for txt in self.dir.glob("*.txt"):
            txt.unlink()
        for p in self.dir.glob("*.jpg"):
            p.unlink()
        for p in self.dir.glob("*.png"):
            p.unlink()
        log = self.dir / "yolo.log" if not eval else self.dir / "eval.log"
        log.write_text(
            f"{config.get('task')}\n Training results: {train_results} \n Validation metrics: {metrics}"
        )
