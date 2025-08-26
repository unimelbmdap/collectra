import shutil, os, yaml, zipfile, tempfile, platform
from rich import print
from abc import ABC, abstractmethod
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime
from .utils import get_all_files
from .utils import processing_msg, error_msg


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


class YOLOModel(Model):
    """
    This class provides a concrete implementation of the Model abstract class
    for YOLO-based object detection models. It handles training, validation,
    detection, and data preprocessing specific to YOLO models.

    Attributes:
        DEFAULT_CONFIG (dict): Default configuration for YOLO training including
                             epochs, image size, verbosity, and device settings
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
        self.dir = (
            Path(config.get("output_log", "logs"))
            / f"{config.get('task')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        print(processing_msg(f"Setting up training environment in {self.dir}"))
        self.dir.mkdir(parents=True, exist_ok=True)

    def train(self, config: dict = {}) -> Path:
        """Train the YOLO model with given configuration."""
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
        if not data:
            raise ValueError(
                error_msg("Input data path is required for training. Found none")
            )
        file_format = config.get("file_format", None)
        if not file_format:
            raise ValueError(
                error_msg("File format is required for training. Found none")
            )
        file_format = file_format.replace(".", "")
        self.preprocess(data, inputs, classes, file_format=file_format)

    def _prepare_assets(
        self, file: Path, is_file: bool = True, inputs: list[str] = []
    ) -> dict:
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
        return f"{item['class_id']} {item['x_center']} {item['y_center']} {item['width_relative']} {item['height_relative']}\n"

    def _prepare_annotations(
        self, data_files: list[dict], classes: list[str]
    ) -> tuple[list[str], list[str]]:
        train_files = []
        val_files = []
        for data in data_files:
            for_validation = False
            image = None
            annotation_str = ""
            for key, value in data.items():
                if key == "collectra_results_metadata":
                    for_validation = value.get("validation", False)
                if value.get("type", None) == "Image":
                    image = value.get("path", None)
                if value.get("type", None) == "ImageCrop":
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
        file_format: str,
        validation: bool = False,
    ) -> None:
        files: list[Path] = get_all_files(data_paths, file_format)
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
        """
        Validate the YOLO model performance.
        """
        merged_config = {**self.config, **config}
        self._setup_environment(merged_config)
        print(f"[bold green]Training object detection model[/bold green]: {self.name}")
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

    def detect(self, config: dict = {}) -> list[dict]:
        """
        Run object detection inference on provided data.

        This method performs object detection on input images or video
        using the trained YOLO model. Results typically include bounding
        boxes, confidence scores, and class predictions.

        Args:
            data (Path): Path to the input data (images/video) for detection
        """
        print(f"[bold green]Running object detection[/bold green]: {self.name}")
        images: list[dict] = []
        image_paths: list[Path] = []
        for image in tqdm(config.get("images", []), desc="Collecting images"):
            image_path = Path(image)
            # if image_path.is_dir():
            #     for img_file in tqdm(image_path.glob("*"), desc=f"Processing directory {image_path}"):
            #         results.append({
            #             "image": img_file,
            #             "results": self.model(img_file)
            #         })
            # if image_path.is_file():
            #     results.append({
            #         "image": image,
            #         "results": self.model(image)
            #     })
            if image_path.is_dir():
                for img_file in tqdm(
                    image_path.glob("*"), desc=f"Processing directory {image_path}"
                ):
                    if img_file.suffix.lower() not in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tiff",
                    ]:
                        continue
                    image_paths.append(str(img_file))
                    images.append({"image": img_file, "results": None})
            if image_path.is_file() and image_path.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
            ]:
                image_paths.append(str(image_path))
                images.append({"image": image_path, "results": None})

        results = self.model(image_paths)

        for index in range(len(results)):
            images[index]["results"] = results[index]

        return images

    def cluster(self, config) -> None:
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
        params = {k: v for d in config.get("params", list()) for k, v in d.items()}
        params["data"] = Path(f"{self.dir}/{self.yolo_config_path}")
        params["project"] = self.dir
        if platform.system() == "Darwin":
            params["device"] = "mps"
        if params.get("epochs", None) is None:
            params["epochs"] = 1
        results = self.model.train(**params)
        return results

    def _execute_validation(self, config: dict):
        """
        Execute the YOLO model validation process.
        """
        print(f"[bold green]Validating YOLO model[/bold green]: {self.name}")
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
        log = self.dir / "yolo.log" if not eval else self.dir / "eval.log"
        log.write_text(
            f"{config.get('task')}\n Training results: {train_results} \n Validation metrics: {metrics}"
        )


class DETECTRON2Engine(Model):
    """
    DETECTRON2 object detection engine implementation (placeholder).

    This class provides a placeholder implementation for Facebook's
    Detectron2 object detection framework. It defines the interface
    but contains placeholder implementations that need to be completed.
    """

    def __init__(self, name: str | Path = "detectron2.pt", config: dict = {}):
        """
        Initialize a DETECTRON2Engine instance.

        Args:
            name (str | Path, optional): Path to the Detectron2 model file.
                                       Defaults to "detectron2.pt".
            config (dict, optional): Configuration parameters for the engine.
                                    Defaults to empty dict.
        """
        super().__init__(name, config)
        self.type = "detectron2"

    def train(self, config: dict = {}) -> Path | None:
        """
        Train the DETECTRON2 model (placeholder implementation).

        This method provides a placeholder for Detectron2 model training.
        The actual implementation should include data loading, model
        configuration, training loop, and checkpoint saving.

        Args:
            config (dict, optional): Training configuration parameters.
                                   Defaults to empty dict.

        Returns:
            Path | None: Path to the trained model or None if training failed
        """
        print(f"[bold green]Training DETECTRON2 model[/bold green]: {self.name}")
        # TODO: Implement DETECTRON2 training logic here
        return None

    def val(self) -> None:
        """
        Validate the DETECTRON2 model (placeholder implementation).

        This method should implement validation logic for Detectron2 models,
        including evaluation metrics calculation and performance assessment.
        """
        print(f"[bold green]Validating DETECTRON2 model[/bold green]: {self.name}")
        # TODO: Implement Detectron2 validation logic here
        pass

    def detect(self, data: Path) -> None:
        """
        Run object detection with DETECTRON2 model (placeholder implementation).

        This method should implement inference logic for Detectron2 models,
        including image preprocessing, model inference, and result processing.

        Args:
            data (Path): Path to the input data for detection
        """
        print(f"[bold green]Running DETECTRON2 detection[/bold green]: {self.name}")
        # TODO: Implement Detectron2 detection logic here
        pass

    def preprocess(self, data: Path, file_format: str) -> None:
        """
        Preprocess data for DETECTRON2 training (placeholder implementation).

        This method should implement data preprocessing specific to Detectron2,
        including format conversion, annotation processing, and dataset preparation.

        Args:
            data (Path): Path to the input data
            file_format (str): Format of the input files
        """
        print(f"[bold green]Preprocessing data for DETECTRON2[/bold green]: {data}")
        # TODO: Implement Detectron2 preprocessing logic here
        pass


class ImageClassifier(Model):
    """
    Image classification engine implementation (placeholder).

    This class provides a placeholder implementation for image classification
    models. It defines the interface for training, validation, and inference
    with image classification models but contains placeholder implementations
    that need to be completed with specific framework logic.
    """

    def __init__(self, name: str | Path = "imageclassifier.pt", config: dict = {}):
        """
        Initialize an ImageClassifier instance.

        Args:
            name (str | Path, optional): Path to the image classifier model file.
                                       Defaults to "imageclassifier.pt".
            config (dict, optional): Configuration parameters for the classifier.
                                    Defaults to empty dict.
        """
        super().__init__(name, config)
        self.type = "image_classifier"

    def detect(self, data: Path) -> None:
        """
        Run image classification on provided data (placeholder implementation).

        This method should implement image classification inference logic,
        including image preprocessing, model prediction, and result formatting.
        Results typically include class predictions and confidence scores.

        Args:
            data (Path): Path to the input images for classification
        """
        print(f"[bold green]Running image classifier[/bold green]: {self.name}")
        # TODO: Implement image classification inference logic
        pass

    def train(self, config: dict = {}) -> Path | None:
        """
        Train the image classifier model (placeholder implementation).

        This method should implement the training pipeline for image classification,
        including data loading, model architecture setup, training loop,
        and model checkpoint saving.

        Args:
            config (dict, optional): Training configuration parameters.
                                   Defaults to empty dict.

        Returns:
            Path | None: Path to the trained model or None if training failed
        """
        print(f"[bold green]Training image classifier[/bold green]: {self.name}")
        # TODO: Implement image classifier training logic
        return None

    def val(self) -> None:
        """
        Validate the image classifier model (placeholder implementation).

        This method should implement validation logic for image classification
        models, including accuracy calculation, confusion matrix generation,
        and other relevant classification metrics.
        """
        print(f"[bold green]Validating image classifier[/bold green]: {self.name}")
        # TODO: Implement image classifier validation logic
        pass

    def preprocess(self, data: Path, file_format: str) -> None:
        """
        Preprocess data for image classification (placeholder implementation).

        This method should implement data preprocessing specific to image
        classification, including image resizing, normalization, data
        augmentation, and dataset organization.

        Args:
            data (Path): Path to the input data
            file_format (str): Format of the input image files
        """
        print(
            f"[bold green]Preprocessing data for image classification[/bold green]: {data}"
        )
        # TODO: Implement image classification preprocessing logic
        pass
