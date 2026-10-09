"""Exercise the real RF-DETR transforms with multiview TIFF pixels and boxes."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tifffile

pytest.importorskip("rfdetr")
from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets._torchvision import Compose, RandomHorizontalFlip, Resize
from rfdetr.datasets.transforms import Normalize
from torchvision.transforms.v2 import ToDtype, ToImage

from collectra import Image
from collectra.tasks.object_detection.rfdetr import ObjectDetectionRFDETR
from collectra.tasks.object_detection.rfdetr_data import _adapt_dataset


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("planar", [False, True])
def test_tiff_train_predict_pixels_and_boxes_match(tmp_path, dtype, planar):
    # Distinct channels and spatial patterns catch lost/reordered channels.
    pixels = np.arange(8 * 10 * 15).reshape(8, 10, 15) % 251
    if dtype == np.float32:
        pixels = pixels.astype(dtype) / 255
    elif dtype == np.uint16:
        pixels = pixels.astype(dtype) * 257
    else:
        pixels = pixels.astype(dtype)
    path = tmp_path / "views.tiff"
    tifffile.imwrite(path, pixels.transpose(2, 0, 1) if planar else pixels,
                     photometric="minisblack", metadata={"axes": "CYX" if planar else "YXC"})
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({
        "images": [{"id": 1, "file_name": path.name, "width": 10, "height": 8}],
        "categories": [{"id": 0, "name": "object"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                         "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0}],
    }))
    transforms = Compose([RandomHorizontalFlip(p=1), Resize((16, 20)),
                          ToImage(), ToDtype(torch.float32, scale=True), Normalize()])
    dataset = _adapt_dataset(CocoDetection(tmp_path, annotations, transforms))
    actual, target = dataset[0]
    task = ObjectDetectionRFDETR(name="test")
    task.model = SimpleNamespace(model_config=SimpleNamespace(num_channels=15))
    raw = task._tiff_tensor(Image(name="views", data=path))
    expected = torch.nn.functional.interpolate(raw.flip(-1)[None], size=(16, 20),
                                               mode="bilinear", align_corners=False,
                                               antialias=True)[0]
    # uint8 training resizes before converting to float, with byte rounding.
    tolerance = 1 / 255 / 0.224 + 1e-6 if dtype == np.uint8 else 1e-6
    mean = torch.tensor([0.485, 0.456, 0.406] * 5)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225] * 5)[:, None, None]
    torch.testing.assert_close(actual, (expected - mean) / std, atol=tolerance, rtol=0)
    torch.testing.assert_close(target["boxes"], torch.tensor([[0.75, 0.5, 0.3, 0.5]]))
    assert target["labels"].tolist() == [0]


@pytest.mark.parametrize("backend", ["cpu", "auto", "gpu", "torchvision"])
def test_tiff_training_uses_supported_backend_and_best_total(tmp_path, monkeypatch, backend):
    task = ObjectDetectionRFDETR(name="test")
    captured = {}
    log = tmp_path / "run"
    weights = log / "weights"
    weights.mkdir(parents=True)
    (weights / "checkpoint_best_regular.pth").write_bytes(b"regular")
    (weights / "checkpoint_best_total.pth").write_bytes(b"ema winner")
    task.model = SimpleNamespace(train=lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(task, "_write_coco_dataset", lambda *args: None)
    monkeypatch.setattr(task, "_reload", lambda: None)
    task._train_fold([{"image_path": "views.tiff"}], [], ["object"], log,
                     {"augmentation_backend": backend})
    assert captured["augmentation_backend"] == "torchvision"
    assert (weights / "best.pt").read_bytes() == b"ema winner"


def test_adapter_preserves_rgb_for_repeated_views_and_averages_distinct_views():
    from collectra.tasks.object_detection.rfdetr_input_adaptation import _expand_model

    conv = torch.nn.Conv2d(3, 8, kernel_size=4, stride=4)
    embeddings = SimpleNamespace(projection=conv, num_channels=3)
    encoder = SimpleNamespace(embeddings=SimpleNamespace(patch_embeddings=embeddings),
                              config=SimpleNamespace(num_channels=3))
    detector = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(backbone=[SimpleNamespace(
            encoder=SimpleNamespace(encoder=encoder))]), args=SimpleNamespace(num_channels=3)),
        model_config=SimpleNamespace(num_channels=3),
        means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225])
    _expand_model(detector, 5)
    views = torch.randn(2, 5, 3, 8, 8)
    torch.testing.assert_close(embeddings.projection(views.flatten(1, 2)), conv(views.mean(1)))
    rgb = views[:, 0]
    torch.testing.assert_close(embeddings.projection(rgb.repeat(1, 5, 1, 1)), conv(rgb))
    assert embeddings.projection.weight.requires_grad
    assert embeddings.num_channels == encoder.config.num_channels == 15
