from types import SimpleNamespace

import torch

from collectra import Image, ImageCrop, ObjectDetectionDETR
from collectra.tasks.object_detection.detr import DetectionTrainResult


def one_crop(image: Image) -> ImageCrop:
    crop = image.make_crop(0.5, 0.5, 0.25, 0.25, name="human")
    crop.partition = "true"
    crop.add_source_parent(image)
    return crop


def test_train_detr_temp_dir(image, tmp_path, monkeypatch):
    """Exercise DETR training orchestration without training Transformers."""
    task = ObjectDetectionDETR(name="label-detector", model="unused")
    crop = one_crop(image)
    seen = {}
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


class FakeProcessor:
    def __call__(self, **kwargs):
        return {"pixel_values": torch.zeros((1, 3, 8, 8))}

    def post_process_object_detection(self, outputs, threshold, target_sizes):
        return [
            {"boxes": torch.tensor([[2.0, 3.0, 8.0, 9.0]]), "labels": torch.tensor([0])}
        ]


class FakeDetrModel(torch.nn.Module):
    def forward(self, **kwargs):
        return SimpleNamespace()


def test_run_detr(image, monkeypatch):
    """Exercise conversion of backend boxes into Collectra ImageCrops."""
    task = ObjectDetectionDETR(name="label-detector", model=FakeDetrModel())
    task._processor = FakeProcessor()
    task._device = "cpu"
    task._categories = ["human"]
    monkeypatch.setattr(task, "_init_model", lambda: None)

    detections = task.run(image)

    assert len(detections) == 1
    assert isinstance(detections[0], ImageCrop)
    assert detections[0].name == "human"
