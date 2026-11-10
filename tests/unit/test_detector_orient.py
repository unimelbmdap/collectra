"""Object detectors can orient the crops of chosen classes with an ImageOrienter.

``orient`` lists the output classes whose crops are oriented. ``orienter_model``
is the orientation checkpoint; without it the published default model is used.
"""

from pathlib import Path

import pytest
from PIL import Image as PILImage

from collectra import Image, ImageCrop, ObjectDetectorRFDETR
from collectra.tasks.machine_learning import orienters
from collectra.tasks.machine_learning.orienters import (
    DEFAULT_ORIENTER_FILE,
    DEFAULT_ORIENTER_REPO,
    ImageOrienter,
    default_orienter_checkpoint,
)
from collectra.types.images import Orientation


def fake_orienter(monkeypatch, *orientations: Orientation) -> list[list[str]]:
    """Make every ImageOrienter predict ``orientations`` in turn, without a model.

    Returns the names of the crops in each batch the orienter was given.
    """
    batches: list[list[str]] = []
    answers = iter(orientations)
    monkeypatch.setattr(ImageOrienter, "_init_model", lambda self: None)

    def predict(self, images):
        batches.append([image.name for image in images])
        return [next(answers) for _ in images]

    monkeypatch.setattr(ImageOrienter, "_predict_batch", predict)
    return batches


@pytest.fixture
def page(tmp_path) -> Image:
    path = tmp_path / "page.png"
    PILImage.new("RGB", (100, 80), "white").save(path)
    return Image(name="page", data=path)


def crops(page: Image, *names: str, orientation=Orientation.NORTH) -> list[ImageCrop]:
    return [
        page.make_crop(0.5, 0.5, 0.2, 0.2, name=name, orientation=orientation)
        for name in names
    ]


def detector(**settings) -> ObjectDetectorRFDETR:
    return ObjectDetectorRFDETR("detect", **settings)


def test_by_default_no_class_is_oriented_and_no_orienter_is_loaded(page, monkeypatch):
    def fail(self):
        raise AssertionError("the orienter must not be loaded")

    monkeypatch.setattr(ImageOrienter, "_init_model", fail)
    task = detector()
    assert task.orient == [] and task.orienter_model is None
    found = crops(page, "label", "stamp")
    assert task._orient_crops(found) == found
    assert {crop.orientation for crop in found} == {Orientation.NORTH}


def test_only_the_listed_classes_are_oriented_in_one_batch(page, monkeypatch):
    batches = fake_orienter(monkeypatch, Orientation.WEST, Orientation.SOUTH)
    task = detector(output=["label", "stamp", "scale"], orient=["label", "scale"])
    label, stamp, scale = task._orient_crops(crops(page, "label", "stamp", "scale"))
    assert batches == [["label", "scale"]]
    assert label.orientation == Orientation.WEST
    assert stamp.orientation == Orientation.NORTH
    assert scale.orientation == Orientation.SOUTH


def test_no_detections_of_the_listed_classes_loads_no_orienter(page, monkeypatch):
    batches = fake_orienter(monkeypatch)
    task = detector(output=["label", "stamp"], orient=["label"])
    task._orient_crops(crops(page, "stamp"))
    assert batches == [] and task._orienter is None


def test_many_crops_are_oriented_in_batches(page, monkeypatch):
    batches = fake_orienter(monkeypatch, *[Orientation.EAST] * 20)
    task = detector(output=["label"], orient=["label"])
    task._orient_crops(crops(page, *["label"] * 20))
    assert [len(batch) for batch in batches] == [16, 4]


def test_the_prediction_turns_the_orientation_the_crop_already_has(page, monkeypatch):
    # The orienter sees the crop as already turned, so its answer adds to that.
    fake_orienter(monkeypatch, Orientation.WEST)
    task = detector(output=["label"], orient=["label"])
    (crop,) = task._orient_crops(crops(page, "label", orientation=Orientation.SOUTH))
    assert crop.orientation == Orientation.EAST


def test_a_single_class_name_is_accepted():
    assert detector(output=["label"], orient="label").orient == ["label"]


def test_orienting_a_class_that_is_not_an_output_is_an_error():
    with pytest.raises(ValueError, match="swatch.*label, stamp"):
        detector(output=["label", "stamp"], orient=["label", "swatch"])


def test_the_orienter_uses_the_given_model_relative_to_the_pipeline(
    page, monkeypatch, tmp_path
):
    from collectra import Pipeline

    fake_orienter(monkeypatch, Orientation.NORTH, Orientation.NORTH)
    pipeline = Pipeline(
        "Detect",
        "det",
        "0.1.0",
        path=str(tmp_path),
        detect={
            "type": "collectra.ObjectDetectorRFDETR",
            "input": ["page"],
            "output": ["label"],
            "orient": ["label"],
            "orienter_model": "models/orient.pth",
        },
    )
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("detect").get_task()
    task._orient_crops(crops(page, "label"))
    assert task._orienter.model == tmp_path / "models" / "orient.pth"
    # One orienter serves every image.
    first = task._orienter
    task._orient_crops(crops(page, "label"))
    assert task._orienter is first


def test_an_absolute_or_missing_orienter_model_is_passed_on(page, monkeypatch):
    fake_orienter(monkeypatch, Orientation.NORTH, Orientation.NORTH)
    task = detector(output=["label"], orient=["label"], orienter_model="/m/o.pth")
    task._orient_crops(crops(page, "label"))
    assert task._orienter.model == Path("/m/o.pth")
    task = detector(output=["label"], orient=["label"])
    task._orient_crops(crops(page, "label"))
    assert task._orienter.model is None


def test_the_default_orienter_model_is_the_published_checkpoint(monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: calls.append(kwargs) or "/cache/orient.pth",
    )
    assert default_orienter_checkpoint() == Path("/cache/orient.pth")
    assert calls == [
        {"repo_id": DEFAULT_ORIENTER_REPO, "filename": DEFAULT_ORIENTER_FILE}
    ]


def test_an_image_orienter_without_a_model_loads_the_default(monkeypatch, tmp_path):
    import torch
    from torchvision import models

    network = models.efficientnet_v2_s(weights=None)
    network.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2), torch.nn.Linear(network.classifier[1].in_features, 4)
    )
    checkpoint = tmp_path / "default.pth"
    torch.save(network.state_dict(), checkpoint)
    monkeypatch.setattr(orienters, "default_orienter_checkpoint", lambda: checkpoint)
    task = ImageOrienter("orient")
    task._init_model()
    assert isinstance(task.model, models.EfficientNet)
    assert not task.model.training
