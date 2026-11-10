"""ObjectDetectorSam3: zero-shot object detection from text prompts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image as PILImage

from collectra import Image, ImageCrop, ObjectDetectorSam3
from collectra.tasks.image_segmentation.sam3 import Sam3Backend
from collectra.tasks.object_detection.base import ObjectDetector


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "source.png"
    PILImage.new("RGB", (100, 50), "white").save(path)
    return Image(name="image", id="image", data=path)


def prediction(boxes, scores):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
    }


def detector(predictions, **settings):
    """A detector whose SAM 3 processor answers each prompt from ``predictions``."""
    task = ObjectDetectorSam3("detect", device="cpu", **settings)
    task._processor = Mock()
    task._processor.set_text_prompt.side_effect = lambda prompt, state: predictions[
        prompt
    ]
    return task


def test_it_is_an_object_detector():
    assert issubclass(ObjectDetectorSam3, ObjectDetector)
    assert ObjectDetectorSam3("detect").threshold == 0.5  # SAM 3's own default


def test_each_box_becomes_a_named_crop_with_its_score(image):
    task = detector(
        {"hourglass": prediction([[10, 5, 30, 25], [50, 0, 100, 50]], [0.9, 0.6])},
        prompt="hourglass",
        output=["hourglass"],
    )
    first, second = task.run(image)
    assert isinstance(first, ImageCrop)
    assert first.name == "hourglass"
    assert first.confidence == pytest.approx(0.9)
    assert (first.x_center, first.y_center) == pytest.approx((0.2, 0.3))
    assert (first.width_relative, first.height_relative) == pytest.approx((0.2, 0.4))
    assert (second.x_center, second.width_relative) == pytest.approx((0.75, 0.5))
    assert second.height_relative == pytest.approx(1.0)
    task._processor.set_image.assert_called_once()


def test_each_output_is_prompted_in_turn_with_its_own_prompt(image):
    task = detector(
        {
            "stamp": prediction([[0, 0, 10, 10]], [0.8]),
            "hand written label": prediction([[20, 20, 40, 40]], [0.7]),
        },
        prompts={"stamp": "stamp", "label": "hand written label"},
    )
    assert [crop.name for crop in task.run(image)] == ["stamp", "label"]
    assert task._processor.reset_all_prompts.call_count == 2


def test_the_prompt_defaults_to_the_output_name_without_underscores(image):
    task = detector(
        {"printed label": prediction([[0, 0, 10, 10]], [0.8])},
        output=["printed_label"],
    )
    assert [crop.name for crop in task.run(image)] == ["printed_label"]


def test_one_prompt_can_serve_every_output(image):
    task = detector(
        {"label": prediction([[0, 0, 10, 10]], [0.8])},
        prompt="label",
        output=["a", "b"],
    )
    assert [crop.name for crop in task.run(image)] == ["a", "b"]


def test_boxes_are_clipped_to_the_image_and_empty_ones_dropped(image):
    task = detector(
        {
            "x": prediction(
                [[-10, -10, 20, 20], [90, 40, 130, 70], [60, 10, 60, 30]],
                [0.9, 0.8, 0.7],
            )
        },
        prompt="x",
    )
    crops = task.run(image)
    assert len(crops) == 2
    assert (crops[0].x_center, crops[0].width_relative) == pytest.approx((0.1, 0.2))
    assert (crops[1].x_center, crops[1].width_relative) == pytest.approx((0.95, 0.1))


def test_no_detections_is_an_empty_list(image):
    assert detector({"x": prediction([], [])}, prompt="x").run(image) == []


def test_the_threshold_of_this_run_is_given_to_the_processor(image):
    task = detector({"x": prediction([], [])}, prompt="x", threshold=0.3)
    task.run(image)
    task._processor.set_confidence_threshold.assert_called_once_with(0.3)


def test_the_orientation_of_the_image_is_kept(image):
    from collectra.types.images import Orientation

    image.orientation = Orientation.EAST
    task = detector({"x": prediction([[0, 0, 10, 10]], [0.9])}, prompt="x")
    assert task.run(image)[0].orientation == Orientation.EAST


def test_inference_runs_under_bfloat16_autocast_and_without_gradients(image):
    seen = []

    def answer(prompt, state):
        seen.append(
            (
                torch.is_autocast_enabled("cpu"),
                torch.get_autocast_dtype("cpu"),
                torch.is_grad_enabled(),
            )
        )
        return {
            "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.bfloat16),
            "scores": torch.tensor([0.9], dtype=torch.bfloat16),
        }

    task = ObjectDetectorSam3("detect", prompt="x", device="cpu")
    task._processor = Mock()
    task._processor.set_text_prompt.side_effect = answer
    assert len(task.run(image)) == 1
    assert seen == [(True, torch.bfloat16, False)]


def test_bad_inputs_are_rejected(image):
    task = detector({}, prompt="x")
    with pytest.raises(ValueError, match="single Image"):
        task.run(image, image)
    with pytest.raises(TypeError, match="Image"):
        task.run("not an image")


def test_empty_prompts_are_rejected(image):
    with pytest.raises(ValueError, match="nonempty"):
        detector({}, prompts={"label": " "}).run(image)


def test_the_model_is_loaded_once_through_the_shared_loader(image, monkeypatch):
    loaded = []
    processor = SimpleNamespace(
        set_image=lambda pil: {},
        reset_all_prompts=lambda state: None,
        set_confidence_threshold=lambda value: None,
        set_text_prompt=lambda prompt, state: prediction([], []),
    )
    monkeypatch.setattr(
        Sam3Backend, "_load_processor", lambda task: loaded.append(task) or processor
    )
    task = ObjectDetectorSam3("detect", prompt="x", device="cpu")
    task.run(image)
    task.run(image)
    assert loaded == [task]


def test_a_pipeline_builds_it_from_its_settings(tmp_path):
    from collectra import Pipeline

    PILImage.new("RGB", (4, 4)).save(tmp_path / "x.png")
    pipeline = Pipeline(
        "Find",
        "fnd",
        "0.1.0",
        path=str(tmp_path),
        page={"type": "collectra.Image", "data": "x.png"},
        detect={
            "type": "collectra.ObjectDetectorSam3",
            "prompts": {"hourglass": "an hourglass"},
            "threshold": 0.4,
            "input": ["page"],
            "output": ["hourglass"],
        },
    )
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("detect").get_task()
    assert isinstance(task, ObjectDetectorSam3)
    assert task._prompts() == {"hourglass": "an hourglass"}
    assert task.threshold == 0.4
    assert task.input_type() is Image and task.output_type() is ImageCrop
    assert "_processor" in task.attributes_to_ignore()
