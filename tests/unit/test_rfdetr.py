from types import SimpleNamespace

import numpy as np
import pytest

from collectra import Image, ImageCrop, ObjectDetectionRFDETR
from collectra.tasks.object_detection.detr import DetectionTrainResult


def one_crop(image: Image) -> ImageCrop:
    crop = image.make_crop(0.5, 0.5, 0.25, 0.25, name="human")
    crop.partition = "true"
    crop.add_source_parent(image)
    return crop


def test_train_rfdetr_temp_dir(image, tmp_path, monkeypatch):
    """Exercise RF-DETR training orchestration without training RF-DETR."""
    task = ObjectDetectionRFDETR(name="label-detector")
    crop = one_crop(image)
    seen = {}
    monkeypatch.setattr(task, "_init_model", lambda: None)
    monkeypatch.setattr(task, "_prepare_assets", lambda *args: ([{"train": 1}], []))
    monkeypatch.setattr(task, "_check_distribution", lambda *args: None)

    def fake_train_fold(train, validation, classes, log, kwargs):
        seen.update(train=train, validation=validation, classes=classes)
        weights = log / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").touch()
        return DetectionTrainResult(log, {"status": "completed"})

    monkeypatch.setattr(task, "_train_fold", fake_train_fold)
    result = task._train(
        crop,
        log="run",
        base_folder=tmp_path,
        classes=["human"],
        validation="true",
        batch=1,
        wandb=False,
    )

    assert result.results_dict == {"status": "completed"}
    assert (result.save_dir / "weights" / "best.pt").exists()
    assert seen == {
        "train": [{"train": 1}],
        "validation": [],
        "classes": ["human"],
    }


class FakeRFDETRModel:
    def predict(self, image, threshold):
        return SimpleNamespace(
            xyxy=np.array([[2.0, 3.0, 8.0, 9.0]]),
            class_id=np.array([0]),
        )


def test_run_rfdetr(image, monkeypatch):
    """Exercise conversion of RF-DETR predictions into ImageCrops."""
    task = ObjectDetectionRFDETR(name="label-detector")
    task.model = FakeRFDETRModel()
    task._categories = ["human"]
    monkeypatch.setattr(task, "_init_model", lambda: None)

    detections = task.run(image)

    assert len(detections) == 1
    assert isinstance(detections[0], ImageCrop)
    assert detections[0].name == "human"


def test_rfdetr_train_forwards_regularization_options(monkeypatch):
    captured = {}

    def fake_training_command(task, inputs, trainer, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "collectra.tasks.object_detection.rfdetr.run_training_command",
        fake_training_command,
    )
    task = ObjectDetectionRFDETR(name="label-detector")

    task.train(
        [],
        weight_decay=0.001,
        drop_path=0.1,
        augmentation="conservative",
        augmentation_backend="auto",
        multi_scale=False,
        use_ema=False,
        early_stopping=True,
        early_stopping_patience=20,
        lr_encoder=5e-5,
        warmup_epochs=2,
    )

    assert captured["weight_decay"] == 0.001
    assert captured["drop_path"] == 0.1
    assert captured["augmentation"] == "conservative"
    assert captured["augmentation_backend"] == "auto"
    assert captured["multi_scale"] is False
    assert captured["use_ema"] is False
    assert captured["early_stopping"] is True
    assert captured["early_stopping_patience"] == 20
    assert captured["lr_encoder"] == 5e-5
    assert captured["warmup_epochs"] == 2


def test_rfdetr_prepare_params_includes_regularization_defaults(tmp_path):
    task = ObjectDetectionRFDETR(name="label-detector")

    params = task._prepare_params(output_dir=tmp_path)

    assert params["weight_decay"] == 1e-4
    assert params["drop_path"] == 0.0
    assert params["multi_scale"] is True
    assert params["use_ema"] is True
    assert params["ema_decay"] == 0.993
    assert params["early_stopping_patience"] == 10
    assert params["lr_encoder"] == 1.5e-4
    assert "aug_config" not in params


def test_rfdetr_rejects_unknown_augmentation_preset(tmp_path):
    task = ObjectDetectionRFDETR(name="label-detector")

    with pytest.raises(ValueError, match="augmentation must be one of"):
        task._prepare_params(output_dir=tmp_path, augmentation="surprise")
