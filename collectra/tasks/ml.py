# """Machine Learning task implementations for Collectra workflows.

# This module provides specialized task classes for machine learning operations
# including object detection, image classification, and model training. It
# integrates with various ML frameworks and provides a standardized interface
# for ML workflows.

# The module includes:
#     - Base MachineLearningTask class with common ML operations
#     - ObjectDetectionYOLO class for YOLO-based object detection
#     - Model management and training functionality
#     - Result processing and metadata handling

# Classes:
#     MachineLearningTask: Base class for ML tasks with model management
#     ObjectDetectionYOLO: YOLO-based object detection task implementation
# """

# __all__ = ["MachineLearningTask", "ObjectDetectionYOLO"]

# from pathlib import Path
# from collectra.types.images import Image, ImageCrop
# from collectra.models.base import Model
# from collectra.models.yolo import YOLOModel
# from collectra.tasks.base import Task
# from dataclasses import dataclass
# from types import UnionType
# import tempfile, zipfile, copy, importlib


# @dataclass(kw_only=True)
# class MachineLearningTask(Task):
#     """Base class for machine learning tasks in Collectra workflows.
    
#     Provides common functionality for ML tasks including model loading,
#     training, evaluation, and inference operations. Serves as the foundation
#     for specialized ML task implementations.
    
#     Attributes:
#         VALID_MODEL: Class reference to the valid model type for this task.
#     """

#     VALID_MODEL = None

#     def load(self, model: str | Path) -> Model:
#         """Load a machine learning model for this task.
        
#         Args:
#             model (str | Path): Path to the model file or model identifier.
            
#         Returns:
#             Model: Loaded model instance ready for operations.
            
#         Raises:
#             Exception: If no valid model type is defined or model path is empty.
#         """
#         if not self.VALID_MODEL or not model:
#             raise Exception(
#                 "No valid model type defined for this task or model path is empty."
#             )        
#         return self.VALID_MODEL(model)    

#     def run(self, **kwargs) -> dict:
#         """Execute machine learning inference on the provided inputs.
        
#         Loads the configured model and performs detection/inference operations
#         on the input data using the model's detect method.
        
#         Args:
#             **kwargs: Input data for ML inference, typically images or other
#                      data types supported by the specific ML task.
                     
#         Returns:
#             dict: Detection/inference results from the ML model.
            
#         Raises:
#             ValueError: If no model is configured for the task.
#         """
#         if not self.config.get("model", None):
#             raise ValueError("Model must be set before running the task.")
#         return self.load(self.config.get("model")).detect(**kwargs)  # type: ignore

#     def train(self) -> Path | None:
#         """Train the machine learning model with the configured parameters.
        
#         Initiates model training using the configured model and training parameters.
#         Handles both directory-based and zip-based model loading depending on
#         the configuration.
        
#         Returns:
#             Path | None: Path to the trained model weights, or None if training fails.
            
#         Raises:
#             ValueError: If no model is configured or model path is invalid.
#         """
#         if not self.config.get("model", None):
#             raise ValueError("Model must be set before training the task.")
#         model = Path(self.config.get("model", ""))
#         if not model:
#             raise ValueError("Model path must be set before training the task.")            
#         if self.config.get("as_dir", False):            
#             return self.load(model).train(self.config)
#         else:
#             with tempfile.TemporaryDirectory() as tmpdirname:
#                 with zipfile.ZipFile(model, "r") as zip_ref:
#                     zip_ref.extract(member=model.name, path=tmpdirname)
#                     model_path = Path(tmpdirname) / model.name
#                     return self.load(model_path).train(self.config)  # type: ignore

#     def eval(self) -> None:
#         """Evaluate the machine learning model performance.
        
#         Performs model validation/evaluation using the configured model
#         and evaluation parameters.
        
#         Raises:
#             ValueError: If no model is configured for validation.
#         """
#         if not self.config.get("model", None):
#             raise ValueError("Model must be set before validating the task.")
#         self.load(self.config.get("model")).val(self.config)  # type: ignore

#     def cluster(self) -> None:
#         """Perform clustering analysis on the dataset.
        
#         Placeholder method for clustering functionality. Currently not implemented
#         but reserved for future clustering operations on the model's dataset.
#         """
#         # if not self.model:
#         #   raise ValueError("Model must be set before clustering the task.")
#         # self.model.cluster(self.config)
#         pass

#     def set_model(self, model: str | Path | None) -> None:
#         """Set the model associated with the task.
        
#         Updates the task's model configuration, preserving the previous model
#         path as the old model for reference.
        
#         Args:
#             model (str | Path | None): The model path as a string, Path object,
#                                       or None to clear the model.
#         """
#         if model:
#             self.config["old_model"] = self.config.get("model", "")
#             self.config["model"] = str(model)

#     def get_model(self) -> str:
#         """Get the model associated with the task.
        
#         Returns:
#             str: The current model path as a string, empty string if no model is set.
#         """
#         return str(self.config.get("model", ""))

#     def get_old_model(self) -> str:
#         """Get the previous model associated with the task.
        
#         Returns:
#             str: The previous model path as a string, empty string if no old model exists.
#         """
#         return str(self.config.get("old_model", ""))

# @dataclass(kw_only=True)
# class ObjectDetectionYOLO(MachineLearningTask):
#     """YOLO-based object detection task implementation.
    
#     Specialized ML task for object detection using YOLO models. Processes
#     images and image crops to detect objects and return bounding box
#     coordinates with class predictions.
    
#     Attributes:
#         VALID_MODEL: YOLOModel class for YOLO-specific operations.
#     """
#     VALID_MODEL = YOLOModel

#     def input_type(self) -> UnionType:
#         """Define input types as Image or ImageCrop.
        
#         Returns:
#             UnionType: Union of Image and ImageCrop types.
#         """
#         return Image | ImageCrop
    
#     def output_type(self) -> type:
#         """Define output type as ImageCrop for detected objects.
        
#         Returns:
#             type: ImageCrop type for bounding box results.
#         """
#         return ImageCrop

#     def check_kwargs(self, **kwargs) -> None:
#         """Validate and process input arguments for object detection.
        
#         Processes input arguments by converting dictionary representations
#         to proper Image or ImageCrop objects. Handles both individual objects
#         and lists of objects with metadata.
        
#         Args:
#             **kwargs: Input arguments that may contain image paths, metadata
#                      dictionaries, or pre-constructed Image/ImageCrop objects.
#         """
#         super().check_kwargs(**kwargs)
#         if self.name == 'field_detector':
#             print("Checking kwargs for field_detector")            
#         for key in self.input:
#             value = copy.deepcopy(self.input[key])
#             if isinstance(value, dict) and "path" in value and "type" in value:                
#                 module_name, class_name = value.pop("type").rsplit(".", 1)
#                 cls = getattr(importlib.import_module(module_name), class_name)     
#                 path = value.pop("path")                                
#                 if "items" in value:                    
#                     self.input[key] = []
#                     for item in value["items"]:
#                         self.input[key].append(cls.build(path, **item))
#                 else:
#                     self.input[key] = cls(path, **value)            
#             elif isinstance(value, str) and Path(value).is_file() and Image.is_image_file(Path(str(value))):
#                 self.input[key] = Image(Path(str(value)))                  

#     def run(self, **kwargs):
#         """Execute YOLO object detection on the provided inputs.
        
#         Performs object detection using the YOLO model, processes the results
#         to extract bounding boxes and class predictions, and creates ImageCrop
#         objects for each detected object.
        
#         Args:
#             **kwargs: Input arguments containing Image or ImageCrop objects
#                  or lists thereof for object detection.
                 
#         Raises:
#             ValueError: If input types don't match the expected Image or ImageCrop types.
            
#         Side Effects:
#             Updates self.output with detected objects as ImageCrop instances,
#             organized by object class names.
#         """            
#         # Validate inputs - handle both single instances and lists
#         for value in kwargs.values():
#             if isinstance(value, list):
#                 if not all(isinstance(v, self.input_type()) for v in value):
#                     raise ValueError(f"Invalid input type in list. Expected {self.input_type()}.")
#             elif not isinstance(value, self.input_type()):
#                 print(value)
#                 raise ValueError(f"Invalid input type. Expected {self.input_type()}.")        
    
#         detections = super().run(**kwargs)                             
    
#         for detection_results in detections.values():
#             # Handle both single detection and list of detections
#             if isinstance(detection_results, list):
#                 # Multiple images processed
#                 for result in detection_results:
#                     self._process_single_detection(result)
#             else:
#                 # Single image processed
#                 self._process_single_detection(detection_results)
    
#         # Convert single-item lists to single items for consistency
#         for key in self.output.keys():
#             if self.output[key] and len(self.output[key]) == 1:
#                 self.output[key] = self.output[key][0]

#     def _process_single_detection(self, detection_result: dict) -> None:
#         """Process a single detection result and update output.
        
#         Args:
#             detection_result (dict): Single detection result containing image and results.
#         """
#         image: Image | None = detection_result.get("image", None)
#         classification_results: list = detection_result.get("results", [])
#         if not classification_results:
#             return
            
#         cls_result = classification_results.pop()                       
#         coordinates = cls_result.boxes.xywhn
#         names = [
#             cls_result.names[cls.item()]
#             for cls in cls_result.boxes.cls.int()
#         ]                                                
        
#         for index in range(len(coordinates)):
#             x, y, w, h = coordinates[index]
#             cropped = ImageCrop(
#                 path=image.path, # type: ignore
#                 x_center=float(x),
#                 y_center=float(y),
#                 width_relative=float(w),
#                 height_relative=float(h),
#             )                     
#             if self.output.get(names[index], None) is None:
#                 self.output[names[index]] = [cropped]
#             else:
#                 self.output[names[index]].append(cropped)
        
#         cls_result.save_crop("output")

#     def save(self, output_path: Path, **kwargs) -> tuple[Path, dict, list[Path]]:
#         """Save object detection results to the specified output path.
        
#         Processes the detection output by converting Image and ImageCrop objects
#         to metadata dictionaries, handles file naming, and prepares the results
#         for serialization.
        
#         Args:
#             output_path (Path): Directory path where results should be saved.
#             **kwargs: Additional arguments including:
#                      - keys_to_remove: List of keys to exclude from output
#                      - format: File format extension for output files
                     
#         Returns:
#             tuple[Path, dict, list[Path]]: A tuple containing:
#                 - Path: The result file path where data will be saved
#                 - dict: Processed result dictionary with metadata
#                 - list[Path]: List of image file paths referenced in the results
#         """
        
#         keys_to_remove = kwargs.get("keys_to_remove", [])
#         file_name = self.output.pop("file", "")
#         for key in keys_to_remove:
#             if key in self.output:
#                 self.output.pop(key)        
#         result_dict = dict()
#         file_name = ""
#         image_path = ""
#         for key, value in self.output.items():
#             if isinstance(value, Image | ImageCrop):
#                 value = value.metadata()
#             if isinstance(value, list):
#                 if all(isinstance(v, ImageCrop) for v in value):
#                     image_path = value[0].path
#                     value = ImageCrop.metadata_list(value)                                        
#                 elif len(value) == 1 and isinstance(value[0], Image):                        
#                     image_path = value[0].path
#                     value = value[0].metadata()
#             if not value:
#                 continue
#             result_dict.update({key: value})            
#             if not file_name and isinstance(value, dict) and value.get("path", ""):
#                 file_name = f"{value.get("path")}.{kwargs.pop("format", "")}"
#         if not file_name:
#             file_name = f"output.{kwargs.pop("format", "")}"
#         result_path = output_path / file_name.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")          
#         return result_path, result_dict, [Path(image_path)] if image_path else []
