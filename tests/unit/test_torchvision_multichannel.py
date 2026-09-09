"""Preserve spectral TIFF channels through classifier export and tensor loading."""

import numpy as np
import pytest
import tifffile
from PIL import Image as PILImage

from collectra import Image, ImageClassifierTorchvision, Link, Orientation
from collectra.tasks.image_classifier.torchvision import _load_classifier_image
from collectra.types.images import read_tiff_channels


def preprocessing(size=8):
    return {
        "resize_size": [size],
        "crop_size": [size],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "bilinear",
    }


@pytest.mark.parametrize("axes", ["CYX", "YXC"])
def test_export_preserves_crop_orientation_channels_and_dtype(
    tmp_path, axes, monkeypatch
):
    source = np.arange(6 * 8 * 75, dtype=np.uint16).reshape(6, 8, 75)
    path = tmp_path / "source.tiff"
    tifffile.imwrite(
        path,
        source.transpose(2, 0, 1) if axes == "CYX" else source,
        photometric="minisblack",
        metadata={"axes": axes},
    )
    image = Image(name="source", data=path)
    crop = image.make_crop(0.5, 0.5, 0.5, 0.5, name="sample")
    crop.orientation = Orientation.WEST
    label = Link(name="Pollen", target=crop, partition="training")
    task = ImageClassifierTorchvision("classifier")
    # TIFF export must never enter Pillow, which cannot represent these samples.
    monkeypatch.setattr(
        type(crop), "pil", lambda self: pytest.fail("TIFF entered Pillow")
    )
    train, val = task._prepare_assets(
        ["Pollen"], tmp_path / "run", "validation", "", label
    )
    exported = next(train.rglob("*.tif"))
    left, top, right, bottom = crop.coordinates()
    expected = np.rot90(source[top:bottom, left:right], -1, axes=(0, 1))
    assert read_tiff_channels(exported).dtype == np.uint16
    np.testing.assert_array_equal(read_tiff_channels(exported), expected)
    tensor = _load_classifier_image(exported)
    assert tensor.shape == (75, expected.shape[0], expected.shape[1])
    np.testing.assert_allclose(
        tensor.numpy(), expected.transpose(2, 0, 1) / 65535, rtol=1e-6
    )
    assert not list(val.rglob("*.tif"))
    assert not list(train.rglob("*.png"))


@pytest.mark.parametrize(
    "dtype, value, expected",
    [("uint8", 255, 1.0), ("uint16", 65535, 1.0), ("float32", 7.5, 7.5)],
)
def test_tiff_tensor_scaling(tmp_path, dtype, value, expected):
    path = tmp_path / "pixels.tif"
    tifffile.imwrite(
        path,
        np.full((75, 4, 4), value, dtype=dtype),
        photometric="minisblack",
        metadata={"axes": "CYX"},
    )
    tensor = _load_classifier_image(path)
    assert tensor.shape == (75, 4, 4)
    assert (tensor == expected).all()


def test_spatial_transforms_and_channel_statistics():
    import torch

    task = ImageClassifierTorchvision("classifier")
    task._preprocessing = preprocessing(4)
    pixels = torch.arange(75 * 4 * 4, dtype=torch.float32).reshape(75, 4, 4)
    # RGB statistics must not be repeated across unrelated spectral bands.
    assert torch.equal(task._transforms()(pixels), pixels)
    assert torch.equal(
        task._transforms(training=True, fliplr=1, flipud=1)(pixels), pixels.flip(-1, -2)
    )
    assert torch.equal(task._transforms(fliplr=1, flipud=1)(pixels), pixels)
    task._preprocessing["mean"] = [1.0] * 75
    task._preprocessing["std"] = [2.0] * 75
    assert torch.equal(task._transforms()(pixels), (pixels - 1) / 2)


def test_all_prepared_splits_have_consistent_channels(tmp_path):
    for split, count in [("train", 75), ("val", 74)]:
        directory = tmp_path / split / "Pollen"
        directory.mkdir(parents=True)
        tifffile.imwrite(
            directory / "sample.tif",
            np.zeros((count, 4, 4), dtype="uint8"),
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )
    task = ImageClassifierTorchvision("classifier")
    task._preprocessing = preprocessing(4)
    with pytest.raises(ValueError, match="Inconsistent input channel counts") as error:
        task._make_datasets(tmp_path / "train", tmp_path / "val", {})
    assert "75 channels" in str(error.value)
    assert "74 channels" in str(error.value)
    assert "train/Pollen/sample.tif" in str(error.value)
    assert "val/Pollen/sample.tif" in str(error.value)


def test_training_and_inference_keep_75_channels(tmp_path, monkeypatch):
    import torch
    from torchvision import models

    class ReadyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(75, 4, 1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, pixels):
            assert pixels.shape[1] == 75
            return self.fc(self.conv(pixels).mean(dim=(2, 3)))

    # Simulate the user's separately adapted model factory, including checkpoint reload.
    monkeypatch.setattr(models, "get_model", lambda *args, **kwargs: ReadyModel())
    task = ImageClassifierTorchvision(
        "classifier", model=ReadyModel(), output=["Pollen", "Spore"]
    )
    task._architecture = "ready-model"
    task._categories = ["Pollen", "Spore"]
    task._preprocessing = preprocessing(8)
    images = []
    for index, (name, partition) in enumerate(
        [("Pollen", "train"), ("Spore", "train"), ("Pollen", "val"), ("Spore", "val")]
    ):
        path = tmp_path / f"{index}.tif"
        tifffile.imwrite(
            path,
            np.full((75, 8, 8), index + 1, dtype="float32"),
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )
        images.append(Image(name=name, data=path, partition=partition))
    result = task._train(
        *images,
        base_folder=tmp_path,
        log="run",
        validation="val",
        epochs=1,
        batch=2,
        device="cpu",
        wandb=False,
    )
    assert task._preprocessing["channels"] == 75
    assert task._preprocessing["mean"] == [0.0] * 75
    assert task._preprocessing["std"] == [1.0] * 75
    checkpoint = result.save_dir / "weights" / "best.pt"
    reloaded = ImageClassifierTorchvision(
        "classifier", model=checkpoint, output=["Pollen", "Spore"], device="cpu"
    )
    monkeypatch.setattr(
        Image, "pil", lambda self: pytest.fail("TIFF inference entered Pillow")
    )
    prediction = reloaded.run(images[0])
    assert type(prediction) is Link
    assert prediction.target is images[0]
    assert prediction.name in ["Pollen", "Spore"]
    assert reloaded.model.conv.in_channels == 75


def test_rgb_loader_and_transforms_preserve_existing_behavior(tmp_path):
    import torch
    from torchvision import transforms as T

    path = tmp_path / "rgb.png"
    PILImage.new("RGB", (12, 10), (40, 80, 120)).save(path)
    task = ImageClassifierTorchvision("classifier")
    task._preprocessing = preprocessing(8)
    pixels = _load_classifier_image(path)
    expected = T.Compose(
        [
            T.Resize([8]),
            T.CenterCrop([8]),
            T.ToTensor(),
            T.Normalize(task._preprocessing["mean"], task._preprocessing["std"]),
        ]
    )(pixels)
    assert torch.equal(task._transforms()(pixels), expected)
