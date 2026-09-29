"""Exercise multichannel classification without downloading pretrained models."""

import numpy as np
import pytest
import tifffile
from PIL import Image as PILImage

from collectra import Image, ImageClassifierYOLO, Link, Orientation
from collectra.tasks.image_classifier.yolo import _classification_channels
from collectra.types.images import read_tiff_channels


def write_tiff(path, pixels, axes="CYX"):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, pixels, photometric="minisblack", metadata={"axes": axes})
    return path


@pytest.mark.parametrize("axes", ["CYX", "YXC", "SYX", "YXS"])
def test_tiff_crop_export_and_loading(tmp_path, axes):
    from collectra.tasks.image_classifier.yolo_multichannel import load_channels

    pixels = np.arange(6 * 8 * 75, dtype=np.uint16).reshape(6, 8, 75)
    path = write_tiff(
        tmp_path / "source.tiff",
        pixels.transpose(2, 0, 1) if axes[0] in "CS" else pixels,
        axes,
    )
    image = Image(name="source", data=path)
    crop = image.make_crop(0.5, 0.5, 0.5, 0.5, name="sample")
    crop.orientation = Orientation.WEST
    label = Link(name="Pollen", target=crop, partition="train")
    task = ImageClassifierYOLO("classifier")
    train, _ = task._prepare_assets(["Pollen"], tmp_path / "run", "val", "", label)
    exported = next(train.rglob("*.tiff"))
    left, top, right, bottom = crop.coordinates()
    expected = np.rot90(pixels[top:bottom, left:right], -1, axes=(0, 1))
    np.testing.assert_array_equal(read_tiff_channels(exported), expected)
    np.testing.assert_allclose(
        load_channels(exported), expected.transpose(2, 0, 1) / 65535, rtol=1e-6
    )
    assert _classification_channels(train) == (75, True)


@pytest.mark.parametrize(
    "mode, count, custom", [("RGB", 3, False), ("RGBA", 4, True), ("L", 1, True)]
)
def test_ordinary_image_channels(tmp_path, mode, count, custom):
    from collectra.tasks.image_classifier.yolo_multichannel import load_channels

    path = tmp_path / "sample.png"
    PILImage.new(mode, (8, 8)).save(path)
    assert _classification_channels(tmp_path) == (count, custom)
    assert load_channels(path).shape == (count, 8, 8)


def test_inconsistent_splits_fail_before_training(tmp_path, monkeypatch):
    from ultralytics import YOLO

    # Avoid model construction: only exercise the task's pre-training boundary.
    yolo = object.__new__(YOLO)
    monkeypatch.setattr(YOLO, "train", lambda *a, **k: pytest.fail("training started"))
    task = ImageClassifierYOLO("classifier", model=yolo)
    for split, channels in [("train", 75), ("val", 74)]:
        write_tiff(
            tmp_path / split / "Pollen" / "sample.tif",
            np.zeros((channels, 8, 8), dtype="uint8"),
        )
    with pytest.raises(ValueError, match="Inconsistent input channel counts") as error:
        task._train_fold(
            tmp_path / "train",
            tmp_path / "val",
            ["Pollen"],
            tmp_path,
            {"project": "project", "base_folder": tmp_path},
        )
    assert "train/Pollen/sample.tif: 75 channels" in str(error.value)
    assert "val/Pollen/sample.tif: 74 channels" in str(error.value)


def test_tensor_transforms_scaling_and_class_mapping(tmp_path):
    import torch
    from ultralytics.cfg import get_cfg
    from collectra.tasks.image_classifier.yolo_multichannel import (
        MultichannelClassificationDataset,
        load_channels,
    )

    path = write_tiff(
        tmp_path / "Spore" / "float.tif", np.full((75, 8, 8), 7.5, dtype="float32")
    )
    assert (load_channels(path) == 7.5).all()
    args = get_cfg(
        overrides={
            "imgsz": 8,
            "scale": 0.0,
            "fliplr": 1.0,
            "flipud": 1.0,
            "erasing": 0.0,
        }
    )
    dataset = MultichannelClassificationDataset(
        tmp_path, args, {0: "Pollen", 1: "Spore"}, augment=True
    )
    assert dataset[0]["cls"] == 1  # Missing earlier class must not shift indices.
    pixels = torch.arange(75 * 8 * 8, dtype=torch.float32).reshape(75, 8, 8)
    # RandomResizedCrop can vary aspect ratio, so isolate the flip operations.
    assert torch.equal(
        dataset.torch_transforms.transforms[1:3][0](pixels), pixels.flip(-1)
    )
    assert torch.equal(dataset.torch_transforms.transforms[2](pixels), pixels.flip(-2))
    assert dataset[0]["img"].shape == (75, 8, 8)


@pytest.mark.parametrize("channels, suffix", [(75, ".tif"), (3, ".png"), (3, ".tiff")])
def test_train_validate_reload_and_infer(tmp_path, monkeypatch, channels, suffix):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import callbacks, checks
    from collectra.tasks.image_classifier.yolo_multichannel import (
        MultichannelClassificationTrainer,
    )

    monkeypatch.setattr(checks, "check_pip_update_available", lambda: None)
    monkeypatch.setattr(callbacks, "add_integration_callbacks", lambda *a: None)
    config = tmp_path / "tiny-cls.yaml"
    config.write_text(
        "nc: 2\nchannels: 3\nbackbone:\n  - [-1, 1, Conv, [8, 3, 2]]\nhead:\n  - [-1, 1, Classify, [nc]]\n"
    )
    model = YOLO(config)
    images = []
    for split in ["train", "val"]:
        for index, name in enumerate(["Pollen", "Spore"]):
            path = tmp_path / f"{split}-{name}{suffix}"
            if suffix == ".png":
                PILImage.new("RGB", (32, 32), (40 + index * 80,) * 3).save(path)
            else:
                write_tiff(
                    path, np.full((channels, 32, 32), 40 + index * 80, dtype="uint8")
                )
            images.append(Image(name=name, data=path, partition=split))
    task = ImageClassifierYOLO("classifier", model=model, output=["Pollen", "Spore"])
    train, val = task._prepare_assets(
        ["Pollen", "Spore"], tmp_path / "data", "val", "", *images
    )
    original_params = task._prepare_params

    def params(**kwargs):
        return {
            **original_params(**kwargs),
            "device": "cpu",
            "workers": 0,
            "amp": False,
            "plots": True,
        }

    monkeypatch.setattr(task, "_prepare_params", params)
    result = task._train_fold(
        train,
        val,
        ["Pollen", "Spore"],
        tmp_path / "data",
        {
            "project": str(tmp_path / "runs"),
            "base_folder": tmp_path,
            "epochs": 1,
            "batch": 2,
            "imgsz": 32,
        },
    )
    assert isinstance(model.trainer, MultichannelClassificationTrainer) == (
        suffix != ".png"
    )
    assert model.trainer.data["channels"] == channels
    assert (
        model.trainer.validator.data["channels"] == channels
    )  # Also final standalone validation.
    assert (
        next(
            m for m in model.model.modules() if isinstance(m, torch.nn.Conv2d)
        ).in_channels
        == channels
    )
    assert 0 <= result.top1 <= 1
    best = model.trainer.best
    assert best.exists()
    reloaded = ImageClassifierYOLO("classifier", model=best, output=["Pollen", "Spore"])
    if suffix != ".png":
        monkeypatch.setattr(Image, "pil", lambda *a: pytest.fail("TIFF entered Pillow"))
    for image in [images[0], images[0].make_crop(0.5, 0.5, 0.5, 0.5, name="crop")]:
        prediction = reloaded.run(image)
        assert prediction.name in ["Pollen", "Spore"]
        assert prediction.target is image
    wrong = Image(
        name="wrong",
        data=write_tiff(tmp_path / "wrong.tif", np.zeros((74, 8, 8), dtype="uint8")),
    )
    with pytest.raises(ValueError, match=f"74 image channels.*expects {channels}"):
        reloaded.run(wrong)

    if channels == 3 and suffix == ".tiff":
        # A tensor-trained RGB checkpoint must also accept ordinary RGB files.
        monkeypatch.undo()
        rgb_path = tmp_path / "ordinary.png"
        PILImage.new("RGB", (32, 32)).save(rgb_path)
        assert reloaded.run(Image(name="rgb", data=rgb_path)).name in [
            "Pollen",
            "Spore",
        ]


def test_pretrained_transfer_uses_dataset_channels():
    from copy import deepcopy

    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import ClassificationModel
    from collectra.tasks.image_classifier.yolo_multichannel import (
        MultichannelClassificationTrainer,
    )

    config = {
        "nc": 2,
        "channels": 3,
        "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
        "head": [[-1, 1, "Classify", [2]]],
    }
    weights = ClassificationModel(deepcopy(config), verbose=False)
    trainer = object.__new__(MultichannelClassificationTrainer)
    trainer.data = {"nc": 2, "channels": 75}
    trainer.args = get_cfg()
    trained = trainer.get_model(config, weights, verbose=False)
    assert trained.yaml["channels"] == 75
    assert config["channels"] == 3
    # Ultralytics transfers compatible layers; Collectra does not rewrite weights.
    assert torch.equal(trained.model[-1].linear.weight, weights.model[-1].linear.weight)
