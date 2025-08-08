from __future__ import annotations
from rich import print
from abc import ABC, abstractmethod
from pathlib import Path
from rocrate.rocrate import ROCrate
from rocrate.model import DataEntity
from ultralytics import YOLO
from tqdm import tqdm
import platform
from .utils import get_all_files
from typing import List
from datetime import datetime
import shutil

class EngineEntity(DataEntity):
  """
  Represents an engine entity within a ROCrate structure.
  
  This class extends the DataEntity class to provide specialized
  functionality for machine learning engine metadata storage.
  """
  
  def __init__(self, crate, identifier=None, properties=None):
    """
    Initialize an EngineEntity instance.
    
    Args:
        crate: The ROCrate instance this entity belongs to
        identifier: Optional unique identifier for the entity
        properties: Optional dictionary of properties for the entity
    """
    super(EngineEntity, self).__init__(crate, identifier, properties)
  
  def _empty(self):
    """
    Return the default empty structure for an engine entity.
    
    Returns:
        dict: Default metadata structure for an engine entity
    """
    return {
        "@id": self.id,
        "@type": "Engine",
        "name": "",            
    }  

class TrainingParameters(DataEntity):
  """
  Represents training parameters within a ROCrate structure.
  
  This class stores and manages training configuration parameters
  for machine learning models within a ROCrate metadata framework.
  """
  
  def __init__(self, crate, identifier=None, properties=None):
    """
    Initialize a TrainingParameters instance.
    
    Args:
        crate: The ROCrate instance this entity belongs to
        identifier: Optional unique identifier for the parameters
        properties: Optional dictionary of training parameter properties
    """
    super(TrainingParameters, self).__init__(crate, identifier, properties)
  
  def _empty(self):
    """
    Return the default empty structure for training parameters.
    
    Returns:
        dict: Default metadata structure for training parameters
    """
    return {
        "@id": self.id,
        "@type": "TrainingParameters",                        
    }

class Engine(ABC):
  """
  Abstract base class for all machine learning engines.
  
  This class defines the common interface and behavior for all machine learning
  engines in the Collectra system. It provides a standardized way to train,
  validate, detect, and manage machine learning models.
  
  Attributes:
      DEFAULT_CONFIG (dict): Default configuration parameters for the engine
  """
  
  DEFAULT_CONFIG: dict = {}

  def __init__(self, name: str | Path, config: dict = {}):
    """
    Initialize an Engine instance.
    
    Args:
        name (str | Path): Name or path to the model file
        config (dict, optional): Configuration parameters for the engine.
                               Defaults to empty dict.
    """
    self.name: Path = Path(name)
    self.config: dict = {**self.DEFAULT_CONFIG, **config}
    self.model = None
    self.type = "generic"

  def __str__(self) -> str:
    """
    Return string representation of the engine.
    
    Returns:
        str: String representation showing the engine name
    """
    return f"{self.name}"  

  @abstractmethod
  def to_crate(self, crate: ROCrate) -> EngineEntity:
    """
    Convert the engine to a ROCrate entity for metadata storage.
    
    Args:
        crate (ROCrate): The ROCrate instance to add the engine to
        
    Returns:
        EngineEntity: The created engine entity with metadata
    """
    pass
  
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
    the engine type and return or display validation metrics.
    """
    pass

  @abstractmethod
  def detect(self, data: Path) -> None:
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

class YOLOEngine(Engine):
  """
  This class provides a concrete implementation of the Engine abstract class
  for YOLO-based object detection models. It handles training, validation,
  detection, and data preprocessing specific to YOLO models.
  
  Attributes:
      DEFAULT_CONFIG (dict): Default configuration for YOLO training including
                           epochs, image size, verbosity, and device settings
  """

  DEFAULT_CONFIG: dict = {    
    "epochs": 50,      # Number of training epochs
    "imgsz": 640,      # Input image size for training
    "verbose": True,   # Enable verbose output during training    
  }

  def __init__(self, name: str | Path = "yolo11n.pt", config: dict = {}):
    """
    Initialize a YOLOEngine instance.
    
    Args:
        name (str | Path, optional): Path to the YOLO model file.
                                   Defaults to "yolo11n.pt".
        config (dict, optional): Configuration parameters for the engine.
                                Defaults to empty dict.
    """
    super().__init__(name, config)
    self.model: YOLO = YOLO(self.name, verbose=True)
    self.type: str = "yolo"
    self.yolo_config_path: str = "yolo_config.yml"
    self.dir: Path = None  # Working directory for training files

  def preprocess(self, data: Path, file_format: str, validation: bool = False) -> None:
    """
    Preprocess data for YOLO training from ROCrate format.
    
    This method processes ROCrate files containing images and bounding box
    annotations, converts them to YOLO format, and creates the necessary
    configuration files for training.
    
    Args:
        data (Path): Path to the input data directory or files
        file_format (str): Format of the input files (e.g., 'grapto')
        
    The method performs the following operations:
    1. Extracts all files matching the specified format
    2. Processes ROCrate files to extract class mappings and bounding boxes
    3. Separates files into training and validation sets
    4. Converts bounding box annotations to YOLO format
    5. Creates YOLO configuration files (train.txt, val.txt, yolo_config.yml)
    """
    files = get_all_files(data, file_format)
    train_files = []  # List of training image files
    val_files = []    # List of validation image files
    number_of_classes = 0
    classes = []      # List of class names
    
    # Set default working directory if not already set
    if self.dir is None:
      self.dir = Path("tmp")    
      
    # Process each ROCrate file
    for file in tqdm(files, desc="Processing files for YOLO training"):
      try:
        crate = ROCrate(file)
        
        # Extract class mapping information (only once)
        if number_of_classes == 0:        
          for e in crate.data_entities:
            if e.type == "ClassMapping":
              print(f"[bold green]Found class mapping[/bold green]: {e.get('classes')}")
              classes = e.get("classes") or []
              number_of_classes = len(classes)                    
              break                      
        
        # Process bounding box annotations
        bounding_box = ""        
        for e in crate.data_entities:
          # Extract and categorize image files
          if e.type == "File":
            e.write(Path(self.dir))          
            if e.get("for_validation"):              
              val_files.append(f"./{e.id}")
            else:
              train_files.append(f"./{e.id}")
          # Convert bounding boxes to YOLO format          
          if e.type == "BoundingBox":
            bounding_box += f"{e.get('class_id')} {e.get('x_center')} {e.get('y_center')} {e.get('width_relative')} {e.get('height_relative')}\n"
        
        # Save bounding box annotations in YOLO format
        bounding_box_file = Path(self.dir) / f"{file.stem}.txt"
        bounding_box_file.write_text(bounding_box)                             
      except Exception as e:
        print(f"[bold red]Error processing file[/bold red]: {file} - {e}")
    
    # Create YOLO dataset configuration
    yolo_config = ""
    if not validation:
      yolo_config = f"train: train.txt\nval: val.txt\nnc: {number_of_classes}\nnames: {classes}"        
      Path(f"{self.dir}/train.txt").write_text("\n".join(train_files))
    else:
      val_files.extend(train_files)
      yolo_config = f"val: val.txt\nnc: {number_of_classes}\nnames: {classes}"

    if len(val_files) == 0:
      print("[bold red]Warning - No validation files found[/bold red]. Using training files for validation.")
      val_files = train_files

    Path(f"{self.dir}/val.txt").write_text("\n".join(val_files))    
    Path(f"{self.dir}/{self.yolo_config_path}").write_text(yolo_config)

  def train(self, config: dict = {}) -> Path:
    """Train the YOLO model with given configuration."""
    merged_config = {**self.config, **config}
    self._setup_training_environment(merged_config)
    
    print(f"[bold green]Training object detection model[/bold green]: {self.name}")
    self._prepare_data(merged_config)
    
    if merged_config.get("test", False):
      print("[bold yellow]Test mode enabled[/bold yellow]: Training will not be performed.")
      train_results = "Test mode: No training performed."
      metrics = "Test mode: No metrics available."
      self._save_results(merged_config, train_results, metrics)      
      return None
    
    train_results = self._execute_training(merged_config)
    new_model_path = self._get_model_path()
    metrics = self._validate_model()
    
    self._save_results(merged_config, train_results, metrics)
    return new_model_path
  
  def val(self, config: dict) -> None:
    """
    Validate the YOLO model performance.        
    """    
    merged_config = {**self.config, **config}
    self._setup_training_environment(merged_config)
    
    print(f"[bold green]Training object detection model[/bold green]: {self.name}")
    self._prepare_data(merged_config)
    val_result = "Validation results"
    if merged_config.get("test", False):
      print("[bold yellow]Test mode enabled[/bold yellow]: Validation will not be performed.")
      val_results = "Test mode: No training performed."
      metrics = "Test mode: No metrics available."
      self._save_results(merged_config, val_results, metrics, eval=True)
      return None    
    metrics = val_results = self._execute_validation(merged_config)
    self._save_results(merged_config, val_results, metrics, eval=True)
  
  def _setup_training_environment(self, config: dict) -> None:
    """
    Setup the training environment and create necessary directories.
    
    This method creates a unique timestamped directory for the training
    session to avoid conflicts with concurrent training runs.
    
    Args:
        config (dict): Configuration dictionary containing directory settings
    """
    # Create unique directory with timestamp to avoid conflicts
    self.dir = Path(config.get("tmp_dir", "tmp")) / f"{config.get('task')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(self.dir)
    self.dir.mkdir(parents=True, exist_ok=True)
  
  def _prepare_data(self, config: dict) -> None:
    """
    Prepare and preprocess training data.
    
    Extracts input data path and file format from configuration,
    then calls the preprocessing method to convert data to YOLO format.
    
    Args:
        config (dict): Configuration dictionary containing input settings
    """
    data = config.get("input", "data")
    file_format = config.get("file_format", "grapto").replace(".", "")
    self.preprocess(data, file_format=file_format)   

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
    train_params = {
      "data": Path(f"{self.dir}/{self.yolo_config_path}"),
      "epochs": int(config.get("epochs", self.DEFAULT_CONFIG["epochs"])),      
      "verbose": config.get("verbose", self.DEFAULT_CONFIG["verbose"]),
      "project": self.dir      
    }
    
    # Use MPS device on macOS for GPU acceleration
    if platform.system() == "Darwin":
      train_params["device"] = "mps"
    
    return self.model.train(**train_params)

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
    # Skip saving if no output directory specified
    if not config.get("output"):
      return
    
    # Create output directory
    output = Path(config.get("output"))
    output.mkdir(parents=True, exist_ok=True)        

    # Copy training artifacts to output directory
    train_results1 = Path(self.dir) / "train"
    train_results2 = Path(self.dir) / "train2"
    validation_results = Path(self.dir) / "val"
    log = output / "yolo.log" if not eval else output / "eval.log"

    if train_results1.exists() and train_results2.exists():
      shutil.copytree(train_results1, output / "train", dirs_exist_ok=True)    
      shutil.copytree(train_results2, output / "train2", dirs_exist_ok=True)          
      
    if validation_results.exists():
      shutil.copytree(validation_results, output / "val", dirs_exist_ok=True)      

    log.write_text(f"{config.get('task')}\n Training results: {train_results} \n Validation metrics: {metrics}")
    
  def to_crate(self, crate: ROCrate) -> EngineEntity:
    """
    Convert the YOLO engine to a ROCrate entity for metadata storage.
    
    This method creates a ROCrate representation of the YOLO engine,
    including the model file and training parameters, for use in
    research data management and reproducibility.
    
    Args:
        crate (ROCrate): The ROCrate instance to add the engine to
        
    Returns:
        EngineEntity: The created engine entity with complete metadata
        
    Raises:
        ValueError: If the model file does not exist
    """            
    model_path = Path(self.name)
    
    # Validate model file exists
    if not model_path.exists():
      raise ValueError(f"[bold red]Model file does not exist[/bold red]: {model_path}")
    
    # Add model file to ROCrate with metadata
    engine_crate = crate.add_file(model_path, properties={
      "name": f"{model_path}",    
      "engine_type": self.type  
    })
    engine_crate["name"] = engine_crate.id
    
    # Create training parameters entity
    training_params = TrainingParameters(crate, identifier=f"{engine_crate.id}-training-params", properties=self.DEFAULT_CONFIG)
    
    # Add entities to crate and link them               
    crate.add(engine_crate)     
    crate.add(training_params)
    engine_crate["trainingParameters"] = [training_params]
    
    return engine_crate  

  def detect(self, data: Path) -> None:
    """
    Run object detection inference on provided data.
    
    This method performs object detection on input images or video
    using the trained YOLO model. Results typically include bounding
    boxes, confidence scores, and class predictions.
    
    Args:
        data (Path): Path to the input data (images/video) for detection
    """
    print(f"[bold green]Running object detection[/bold green]: {self.name}")
    # TODO: Implement YOLO detection logic and result handling
    pass

class DETECTRON2Engine(Engine):
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
    
    def to_crate(self, crate: ROCrate) -> EngineEntity:
      """
      Convert DETECTRON2 engine to ROCrate entity (placeholder implementation).
      
      This method should create a ROCrate representation of the Detectron2
      engine including model files, configuration, and metadata.
      
      Args:
          crate (ROCrate): The ROCrate instance to add the engine to
          
      Returns:
          EngineEntity: The created engine entity with metadata
      """
      # TODO: Implement ROCrate conversion for Detectron2
      pass


class ImageClassifier(Engine):
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

    def to_crate(self, crate: ROCrate) -> EngineEntity:
      """
      Convert image classifier to ROCrate entity (placeholder implementation).
      
      This method should create a ROCrate representation of the image
      classification engine including model files, training parameters,
      and classification metadata.
      
      Args:
          crate (ROCrate): The ROCrate instance to add the engine to
          
      Returns:
          EngineEntity: The created engine entity with metadata
      """
      # TODO: Implement ROCrate conversion for image classifier
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
      print(f"[bold green]Preprocessing data for image classification[/bold green]: {data}")
      # TODO: Implement image classification preprocessing logic
      pass

