"""The pipeline's outputs are the source of truth for YOLO classifier class names.

Ultralytics numbers classes by sorted folder name and keeps the folder names in the
model. So training folders are numbered in the pipeline's order, and prediction turns
the class number into the pipeline's output, whatever names the model holds.
"""

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image as PillowImage

from collectra import Image, ImageClassifierYOLO, Link
from collectra.tasks.image_classifier import yolo as yolo_module
from collectra.tasks.image_classifier.yolo import class_folder_names

OUTPUTS = ["stamp", "label", "barcode"]


def test_folders_are_numbered_in_the_given_order():
    assert class_folder_names(OUTPUTS) == ["00_stamp", "01_label", "02_barcode"]


def test_folder_numbers_keep_sorting_in_order_past_ten_classes():
    names = class_folder_names([f"c{index}" for index in range(12)])
    assert sorted(names) == names


def test_images_go_in_the_numbered_folder_of_their_class(tmp_path):
    images = []
    for name in ("label", "stamp"):
        path = tmp_path / f"{name}.png"
        PillowImage.new("RGB", (8, 8)).save(path)
        images.append(Image(name=name, data=path, partition="train"))
    task = ImageClassifierYOLO("classifier", output=["stamp", "label"])
    train, _ = task._prepare_assets(
        ["stamp", "label"], tmp_path / "run", "val", "", *images
    )
    assert sorted(path.name for path in train.iterdir()) == ["00_stamp", "01_label"]
    assert [p.name for p in (train / "00_stamp").iterdir()] == ["stamp.png"]
    assert [p.name for p in (train / "01_label").iterdir()] == ["label.png"]


@pytest.fixture
def fake_yolo(monkeypatch):
    class FakeYOLO:
        model_name = "fake"

    module = ModuleType("ultralytics.models")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics.models", module)
    return FakeYOLO


def train_with(task, fake_yolo, tmp_path, images, monkeypatch, **kwargs):
    seen = {}

    def prepare(classes, *args, **options):
        seen["classes"] = classes
        return "train", "val"

    task.model = fake_yolo()
    monkeypatch.setattr(task, "_init_model", lambda: None)
    monkeypatch.setattr(task, "_prepare_assets", prepare)
    monkeypatch.setattr(task, "_train_fold", lambda *args, **options: None)
    task._train(*images, log="run", base_folder=tmp_path, **kwargs)
    return seen["classes"]


def labelled(tmp_path, *names):
    path = tmp_path / "x.png"
    PillowImage.new("RGB", (8, 8)).save(path)
    image = Image(name="x", data=path)
    return [Link(name=name, target=image) for name in names]


def test_training_uses_the_given_classes_in_their_order(
    tmp_path, fake_yolo, monkeypatch
):
    task = ImageClassifierYOLO("classifier", output=["a", "b"])
    classes = train_with(
        task,
        fake_yolo,
        tmp_path,
        labelled(tmp_path, "a"),
        monkeypatch,
        classes=["z", "a"],
    )
    assert classes == ["z", "a"]


def test_training_uses_the_outputs_when_no_classes_are_given(
    tmp_path, fake_yolo, monkeypatch
):
    task = ImageClassifierYOLO("classifier", output=["z", "a"])
    classes = train_with(
        task, fake_yolo, tmp_path, labelled(tmp_path, "a"), monkeypatch
    )
    assert classes == ["z", "a"]


def test_training_finds_classes_in_the_images_without_either(
    tmp_path, fake_yolo, monkeypatch
):
    task = ImageClassifierYOLO("classifier")
    classes = train_with(
        task, fake_yolo, tmp_path, labelled(tmp_path, "z", "a", "z"), monkeypatch
    )
    assert classes == ["a", "z"]


def classifier(tmp_path, model_names, top1, outputs=("stamp", "label")):
    path = tmp_path / "image.png"
    PillowImage.new("RGB", (8, 8)).save(path)
    task = ImageClassifierYOLO("classifier", model="unused.pt", output=list(outputs))
    task._init_model = lambda: None
    task.model = lambda _: [
        SimpleNamespace(probs=SimpleNamespace(top1=top1), names=model_names)
    ]
    return task, Image(name="image", data=path)


def warnings_from(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_prediction_is_the_output_at_the_class_number(tmp_path, caplog):
    task, image = classifier(tmp_path, {0: "00_stamp", 1: "01_label"}, top1=1)
    with caplog.at_level(logging.WARNING, logger=yolo_module.logger.name):
        assert task.run(image).name == "label"
    assert warnings_from(caplog) == []


def test_older_models_with_plain_names_are_silent_when_they_agree(tmp_path, caplog):
    task, image = classifier(tmp_path, {0: "stamp", 1: "label"}, top1=0)
    with caplog.at_level(logging.WARNING, logger=yolo_module.logger.name):
        assert task.run(image).name == "stamp"
    assert warnings_from(caplog) == []


def test_pipeline_wins_over_different_names_in_the_model(tmp_path, caplog):
    task, image = classifier(tmp_path, {0: "label", 1: "stamp"}, top1=0)
    with caplog.at_level(logging.WARNING, logger=yolo_module.logger.name):
        assert task.run(image).name == "stamp"  # not the model's "label"
        task.run(image)
    (warning,) = warnings_from(caplog)  # once, not for every image
    assert "label, stamp" in warning and "stamp, label" in warning


def test_wrong_number_of_outputs_for_the_model_is_an_error(tmp_path):
    task, image = classifier(tmp_path, {0: "a", 1: "b", 2: "c"}, top1=0)
    with pytest.raises(ValueError, match="2 output.*3 classes"):
        task.run(image)
