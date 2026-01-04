__all__ = ["SuryaOCR", "SuryaLineDetector"]

from dataclasses import dataclass, field

from dotenv import load_dotenv

from collectra.tasks.base import Task
from collectra.types.images import Image, ImageCrop
from collectra.types.texts import Text

load_dotenv()


@dataclass
class SuryaOCR(Task):
    name: str
    # recognition_predictor:"RecognitionPredictor" = field(init=False)
    # detection_predictor:"DetectionPredictor" = field(init=False)

    def __post_init__(self):
        """Initialize the LLM instance and message templates after object creation.

        Loads the specified LLM model using the llmloader library and sets up
        the initial system message for the conversation context.
        """
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        self.detection_predictor = DetectionPredictor()
        foundation_predictor = FoundationPredictor()
        self.recognition_predictor = RecognitionPredictor(foundation_predictor)

    def input_type(self) -> type | tuple:
        """Define the expected input types for this LLM task.

        Returns:
            type | tuple: Tuple of str and Path types for text and file inputs.
        """
        return Image

    def output_type(self) -> type | tuple:
        """Define the expected output types for this LLM task.

        Returns:
            type | tuple: String type for generated text outputs.
        """
        return Text

    def run(self, **kwargs):
        for key, value in kwargs.items():
            assert isinstance(value, Image)
            image = value.pil()
            predictions = self.recognition_predictor(
                [image], det_predictor=self.detection_predictor
            )
        lines = [line.text for line in predictions[0].text_lines]
        text = str("\n".join(lines))
        for output_key in self.output.keys():
            if key in output_key:
                self.output[output_key] = text


@dataclass
class SuryaLineDetector(Task):
    name: str
    merge_horizontal: bool = False
    min_height: int = 0
    # detection_predictor:"DetectionPredictor" = field(init=False)

    def __post_init__(self):
        """Initialize the LLM instance and message templates after object creation.

        Loads the specified LLM model using the llmloader library and sets up
        the initial system message for the conversation context.
        """
        from surya.detection import DetectionPredictor

        self.detection_predictor = DetectionPredictor()
        self.min_height = self.min_height or 0

    def input_type(self) -> type | tuple:
        """Define the expected input types for this LLM task.

        Returns:
            type | tuple: Tuple of str and Path types for text and file inputs.
        """
        return Image

    def output_type(self) -> type | tuple:
        """Define the expected output types for this LLM task.

        Returns:
            type | tuple: String type for generated text outputs.
        """
        return ImageCrop

    # def run(self, image:Image) -> list[ImageCrop]:
    def run(self, **kwargs) -> list[ImageCrop]:
        results = None
        for key, image in kwargs.items():
            # Hack until Task just takes a single input
            if isinstance(image, list):
                for individual_image in image:
                    self.run(**{key: individual_image})
                return

            assert isinstance(image, Image), f"Image {key} is of class {type(image)}"
            pil_image = image.pil()
            predictions = self.detection_predictor([pil_image])

            for output_key in self.output.keys():
                if key in output_key:
                    results = self.output.get(output_key, [])
                    break

            if results is None:
                results = []
            assert results is not None

            # Sort bounding boxes vertically
            bounding_boxes = sorted(
                [polygon_box.bbox for polygon_box in predictions[0].bboxes],
                key=lambda bbox: bbox[1],
            )

            if len(bounding_boxes) == 0 or (
                self.min_height and image.height <= self.min_height
            ):
                crop = image.make_crop(0.5, 0.5, 1.0, 1.0)
                results.append(crop)
            elif self.merge_horizontal:
                bbox = bounding_boxes[0]
                top = bbox[1]
                bottom = bbox[3]

                for bbox in bounding_boxes:
                    if bbox[1] >= bottom:
                        crop = image.make_crop_bounding_box(
                            0, top, image.width, bottom, min_height=self.min_height
                        )
                        results.append(crop)
                        top = bbox[1]
                        bottom = bbox[3]
                    else:
                        top = min(top, bbox[1])
                        bottom = max(bottom, bbox[3])

                crop = image.make_crop_bounding_box(0, top, image.width, bottom)
                results.append(crop)
            else:
                for bbox in bounding_boxes:
                    crop = image.make_crop_bounding_box(
                        *bbox, min_height=self.min_height
                    )
                    results.append(crop)

            for output_key in self.output.keys():
                self.output[output_key] = results
