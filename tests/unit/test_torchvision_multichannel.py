"""Preserve spectral TIFF channels through classifier export and tensor loading."""

import numpy as np
import pytest
import tifffile
from PIL import Image as PILImage

from collectra import Image, ImageClassifierTorchvision, Link, Orientation
from collectra.tasks.image_classifier.torchvision import (
    _adapt_rgb_input_channels,
    _load_classifier_image,
)
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
    # 3-length mean/std cannot normalize a 75-channel tensor directly.
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
            self.conv1 = torch.nn.Conv2d(75, 4, 1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, pixels):
            assert pixels.shape[1] == 75
            return self.fc(self.conv1(pixels).mean(dim=(2, 3)))

    # Simulate the user's separately adapted model factory, including checkpoint reload.
    monkeypatch.setattr(models, "get_model", lambda *args, **kwargs: ReadyModel())
    task = ImageClassifierTorchvision(
        "classifier", model=ReadyModel(), output=["Pollen", "Spore"]
    )
    task._architecture = "resnet18"
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
    assert task._preprocessing["mean"] == preprocessing()["mean"] * 25
    assert task._preprocessing["std"] == preprocessing()["std"] * 25
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
    assert reloaded.model.conv1.in_channels == 75


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


@pytest.mark.parametrize("channels,num_views", [(15, 5), (75, 25)])
def test_multiview_rgb_statistics_repeat_in_order(tmp_path, channels, num_views):
    for split in ("train", "val"):
        directory = tmp_path / split / "Pollen"
        directory.mkdir(parents=True)
        tifffile.imwrite(
            directory / "sample.tif",
            np.zeros((channels, 4, 4), dtype="uint8"),
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )
    task = ImageClassifierTorchvision("classifier")
    task._preprocessing = preprocessing(4)
    base_mean = list(task._preprocessing["mean"])
    base_std = list(task._preprocessing["std"])
    task._make_datasets(tmp_path / "train", tmp_path / "val", {})
    assert task._input_channels == channels
    assert task._preprocessing["mean"] == base_mean * num_views
    assert task._preprocessing["std"] == base_std * num_views


def test_non_divisible_channel_count_uses_identity_normalization(tmp_path):
    for split in ("train", "val"):
        directory = tmp_path / split / "Pollen"
        directory.mkdir(parents=True)
        tifffile.imwrite(
            directory / "sample.tif",
            np.zeros((10, 4, 4), dtype="uint8"),
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )
    task = ImageClassifierTorchvision("classifier")
    task._preprocessing = preprocessing(4)
    task._make_datasets(tmp_path / "train", tmp_path / "val", {})
    assert task._input_channels == 10
    assert task._preprocessing["mean"] == [0.0] * 10
    assert task._preprocessing["std"] == [1.0] * 10


@pytest.mark.parametrize("architecture", ["resnet", "convnext"])
def test_adapt_rgb_input_channels_repeats_and_scales_weights(architecture):
    import torch

    class TinyResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 2, 1, bias=True)

    model = TinyResNet()
    layer_path = "conv1"
    if architecture == "convnext":
        model.features = torch.nn.Sequential(torch.nn.Sequential(model.conv1))
        del model.conv1
        layer_path = "features.0.0"
    conv = model.get_submodule(layer_path)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1))
        conv.bias.copy_(torch.tensor([1.0, -1.0]))
    original = conv.weight.detach().clone()
    _adapt_rgb_input_channels(model, 15)
    adapted = model.get_submodule(layer_path)
    assert adapted.in_channels == 15
    expected = original.repeat(1, 5, 1, 1) / 5
    assert torch.equal(adapted.weight, expected)
    assert torch.equal(adapted.bias, torch.tensor([1.0, -1.0]))
    rgb = torch.rand(1, 3, 4, 4)
    torch.testing.assert_close(adapted(rgb.repeat(1, 5, 1, 1)), conv(rgb))


def test_detected_channels_expand_resnet_conv1_for_training(tmp_path):
    import torch

    class TinyResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, 1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, pixels):
            return self.fc(self.pool(self.conv1(pixels)).flatten(1))

    for split in ("train", "val"):
        directory = tmp_path / split / "Pollen"
        directory.mkdir(parents=True)
        tifffile.imwrite(
            directory / "sample.tif",
            np.zeros((15, 4, 4), dtype="uint8"),
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )

    task = ImageClassifierTorchvision("classifier", model=TinyResNet())
    task._architecture = "resnet18"
    task._preprocessing = preprocessing(4)
    task._make_datasets(tmp_path / "train", tmp_path / "val", {})
    task._prepare_training_model(["Pollen", "Spore"], pretrained=False, freeze=False)
    assert task._input_channels == 15
    assert task.model.conv1.in_channels == 15


def test_checkpoint_load_restores_input_channels(tmp_path, monkeypatch):
    import torch
    from torchvision import models

    class TinyResNet(torch.nn.Module):
        def __init__(self, in_channels=3):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels, 4, 1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, pixels):
            return self.fc(self.pool(self.conv1(pixels)).flatten(1))

    checkpoint_model = TinyResNet(3)
    _adapt_rgb_input_channels(checkpoint_model, 15)
    checkpoint = {
        "architecture": "resnet18",
        "model_kwargs": {},
        "classes": ["Pollen", "Spore"],
        "preprocessing": preprocessing(4),
        "state_dict": checkpoint_model.state_dict(),
        "input_channels": 15,
    }
    path = tmp_path / "multichannel.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr(models, "get_model", lambda *args, **kwargs: TinyResNet(3))
    task = ImageClassifierTorchvision("classifier", model=path)
    task._load()
    assert task._input_channels == 15
    assert task.model.conv1.in_channels == 15


def test_checkpoint_load_without_input_channels_defaults_to_3(tmp_path, monkeypatch):
    import torch
    from torchvision import models

    class TinyResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, 1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, pixels):
            return self.fc(self.pool(self.conv1(pixels)).flatten(1))

    checkpoint_model = TinyResNet()
    checkpoint = {
        "architecture": "resnet18",
        "model_kwargs": {},
        "classes": ["Pollen", "Spore"],
        "preprocessing": preprocessing(4),
        "state_dict": checkpoint_model.state_dict(),
    }
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr(models, "get_model", lambda *args, **kwargs: TinyResNet())
    task = ImageClassifierTorchvision("classifier", model=path)
    task._load()
    assert task._input_channels == 3
    assert task.model.conv1.in_channels == 3


def test_adapt_rgb_input_channels_requires_supported_input_layer():
    import torch

    class NoConv1(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="Could not find first Conv2d"):
        _adapt_rgb_input_channels(NoConv1(), 15)


def test_adapt_rgb_input_channels_requires_conv1_conv2d():
    import torch

    class BadConv1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Linear(3, 4)

    with pytest.raises(ValueError, match="Expected first layer to be Conv2d, got Linear"):
        _adapt_rgb_input_channels(BadConv1(), 15)


def test_unsupported_architecture_fails_for_multichannel_training(tmp_path):
    import torch

    class TinyResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, 1)
            self.fc = torch.nn.Linear(4, 2)

    task = ImageClassifierTorchvision("classifier", model=TinyResNet())
    task._architecture = "efficientnet_b0"
    task._input_channels = 15
    task._categories = ["Pollen", "Spore"]
    with pytest.raises(
        ValueError,
        match="15-channel input adaptation is currently supported only for",
    ):
        task._prepare_training_model(["Pollen", "Spore"], pretrained=False, freeze=False)


def test_unsupported_architecture_fails_for_multichannel_checkpoint_load(
    tmp_path, monkeypatch
):
    import torch
    from torchvision import models

    class TinyResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, 1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, pixels):
            return self.fc(self.pool(self.conv1(pixels)).flatten(1))

    checkpoint = {
        "architecture": "efficientnet_b0",
        "model_kwargs": {},
        "classes": ["Pollen", "Spore"],
        "preprocessing": preprocessing(4),
        "state_dict": TinyResNet().state_dict(),
        "input_channels": 15,
    }
    path = tmp_path / "unsupported.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr(models, "get_model", lambda *args, **kwargs: TinyResNet())
    task = ImageClassifierTorchvision("classifier", model=path)
    with pytest.raises(
        ValueError,
        match="15-channel input adaptation is currently supported only for",
    ):
        task._load()


def test_train_adapts_resnet_conv1_before_first_forward(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from torchvision import models
    from torchvision.transforms import InterpolationMode

    class TinyResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, 1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, pixels):
            return self.fc(self.pool(self.conv1(pixels)).flatten(1))

    weights = SimpleNamespace(
        meta={"categories": [str(index) for index in range(1000)]},
        transforms=lambda: SimpleNamespace(
            crop_size=[8],
            resize_size=[8],
            mean=[0.5, 0.4, 0.3],
            std=[0.2, 0.2, 0.2],
            interpolation=InterpolationMode.BILINEAR,
        ),
    )
    monkeypatch.setattr(models, "get_model", lambda *args, **kwargs: TinyResNet())
    monkeypatch.setattr(
        models, "get_model_weights", lambda name: SimpleNamespace(DEFAULT=weights)
    )

    images = []
    for index, (name, partition) in enumerate(
        [
            ("Pollen", "train"),
            ("Spore", "train"),
            ("Pollen", "val"),
            ("Spore", "val"),
        ]
    ):
        path = tmp_path / f"sample-{index}.tif"
        tifffile.imwrite(
            path,
            np.full((15, 8, 8), index + 1, dtype="float32"),
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )
        images.append(Image(name=name, data=path, partition=partition))

    task = ImageClassifierTorchvision("classifier", model="resnet18")
    result = task._train(
        *images,
        base_folder=tmp_path,
        log="run",
        validation="val",
        epochs=1,
        batch=2,
        workers=0,
        device="cpu",
        wandb=False,
    )
    assert task._input_channels == 15
    assert task.model.conv1.in_channels == 15
    assert result.results_dict["classes"] == ["Pollen", "Spore"]
