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


def test_train_rfdetr_max_items_limits_each_split(image, tmp_path, monkeypatch):
    task = ObjectDetectionRFDETR(name="label-detector")
    seen = {}
    monkeypatch.setattr(task, "_init_model", lambda: None)
    monkeypatch.setattr(
        task,
        "_prepare_assets",
        lambda *args: ([{"train": i} for i in range(5)], [{"val": i} for i in range(4)]),
    )
    monkeypatch.setattr(task, "_check_distribution", lambda *args: None)

    def fake_train_fold(train, validation, classes, log, kwargs):
        seen.update(train=train, validation=validation)
        return DetectionTrainResult(log, {"status": "completed"})

    monkeypatch.setattr(task, "_train_fold", fake_train_fold)
    task._train(
        one_crop(image),
        log="run",
        base_folder=tmp_path,
        classes=["human"],
        max_items=2,
    )

    assert seen == {
        "train": [{"train": 0}, {"train": 1}],
        "validation": [{"val": 0}, {"val": 1}],
    }


class FakeRFDETRModel:
    def predict(self, image, threshold, **kwargs):
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


def test_tiff_tensor_keeps_all_channels(tmp_path):
    import tifffile

    pixels = np.arange(40 * 60 * 5, dtype=np.uint16).reshape(40, 60, 5)
    path = tmp_path / "multi.tif"
    tifffile.imwrite(path, pixels, photometric="minisblack", planarconfig="contig")
    task = ObjectDetectionRFDETR(name="label-detector")
    task.model = SimpleNamespace(model_config=SimpleNamespace(num_channels=5))

    crop = Image(name="sheet", data=path).make_crop(0.5, 0.5, 0.5, 0.5, name="c")
    tensor = task._tiff_tensor(crop)

    left, top, right, bottom = crop.coordinates()
    expected = pixels[top:bottom, left:right].transpose(2, 0, 1) / 65535
    assert tensor.shape == (5, bottom - top, right - left)
    np.testing.assert_allclose(tensor.numpy(), expected, rtol=1e-6)


def test_tiff_tensor_rejects_channel_mismatch(tmp_path):
    import tifffile

    path = tmp_path / "multi.tif"
    tifffile.imwrite(
        path,
        np.zeros((20, 30, 5), dtype=np.uint8),
        photometric="minisblack",
        planarconfig="contig",
    )
    task = ObjectDetectionRFDETR(name="label-detector")
    task.model = SimpleNamespace(model_config=SimpleNamespace(num_channels=3))

    with pytest.raises(ValueError, match="5 image channels"):
        task._tiff_tensor(Image(name="sheet", data=path))


def test_checkpoint_channels_reads_patch_projection(tmp_path):
    import torch

    from collectra.tasks.object_detection.rfdetr_channels import (
        PROJECTION_KEY,
        checkpoint_channels,
    )

    plain = tmp_path / "plain.pth"
    torch.save({"model": {PROJECTION_KEY: torch.zeros(8, 15, 2, 2)}}, plain)
    lightning = tmp_path / "last.ckpt"
    torch.save({"state_dict": {f"model.{PROJECTION_KEY}": torch.zeros(8, 6, 2, 2)}}, lightning)

    assert checkpoint_channels(plain) == 15
    assert checkpoint_channels(lightning) == 6
    assert checkpoint_channels(tmp_path / "missing.pth") == 3


def test_channel_aware_loader_widens_before_and_after_loading(tmp_path):
    import torch

    from collectra.tasks.object_detection.rfdetr_channels import (
        PROJECTION_KEY,
        _channel_aware,
    )

    def model():
        embeddings = SimpleNamespace(
            projection=torch.nn.Conv2d(3, 4, 2, stride=2), num_channels=3
        )
        encoder = SimpleNamespace(
            encoder=SimpleNamespace(embeddings=SimpleNamespace(patch_embeddings=embeddings))
        )
        return SimpleNamespace(backbone=[SimpleNamespace(encoder=encoder)]), embeddings

    seen = []

    def load(nn_model, config):
        seen.append(nn_model.backbone[0].encoder.encoder.embeddings.patch_embeddings.projection.in_channels)

    wide = tmp_path / "wide.pth"
    torch.save({"model": {PROJECTION_KEY: torch.zeros(4, 9, 2, 2)}}, wide)
    nn_model, embeddings = model()
    _channel_aware(load, widen_after=False)(
        nn_model, SimpleNamespace(pretrain_weights=str(wide), num_channels=3)
    )
    assert seen == [9] and embeddings.num_channels == 9

    rgb = tmp_path / "rgb.pth"
    torch.save({"model": {PROJECTION_KEY: torch.zeros(4, 3, 2, 2)}}, rgb)
    nn_model, embeddings = model()
    original = embeddings.projection.weight.detach().clone()
    _channel_aware(load, widen_after=True)(
        nn_model, SimpleNamespace(pretrain_weights=str(rgb), num_channels=6)
    )
    widened = embeddings.projection.weight.detach()
    assert seen[-1] == 3 and widened.shape[1] == 6
    torch.testing.assert_close(widened[:, 3:], original / 2)
