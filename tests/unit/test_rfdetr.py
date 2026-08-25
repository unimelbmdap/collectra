from types import SimpleNamespace

import numpy as np

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
