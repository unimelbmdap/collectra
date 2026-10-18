import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image as PILImage

from collectra import Image, ImageCrop, ImageSegmentation, ImageSegmentationYOLO
from collectra.tasks.image_segmentation import yolo
from collectra.tasks.image_segmentation.data import SegmentationSample, export_dataset


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "page.png"
    PILImage.new("RGB", (32, 32), "white").save(path)
    image = Image(name="page", id="page", data=path)
    pixels = np.zeros((32, 32), dtype="uint8")
    pixels[3:20, 4:24] = 1
    pixels[6:10, 7:12] = 0  # hole
    pixels[26:29, 27:30] = 1  # disconnected island, same instance
    mask = tmp_path / "mask.png"
    PILImage.fromarray(pixels).save(mask)
    instance = ImageSegmentation(name="text_region", data=path, mask=mask, parents=[image.id])
    return SegmentationSample(image, [instance])


def fake_predictor(masks, names=None):
    torch = pytest.importorskip("torch")
    prediction = SimpleNamespace(
        masks=SimpleNamespace(data=torch.tensor(np.asarray(masks))),
        boxes=SimpleNamespace(cls=torch.zeros(len(masks))), names=names or {0: "text_region"},
    )
    return SimpleNamespace(task="segment", predict=Mock(return_value=[prediction]))


@pytest.mark.parametrize("orientation", ["north", "east", "south", "west"])
def test_prediction_preserves_crop_orientation_and_binary_mask(sample, tmp_path, orientation):
    from collectra.tasks.image_segmentation.data import mask_in_view

    image = ImageCrop(name="crop", id="crop", data=sample.image.data, x_center=.5, y_center=.5,
                      width_relative=.5, height_relative=.5, orientation=orientation, partition="test")
    mask = np.asarray(mask_in_view(image, sample.instances[0]._mask_image())) != 0
    task = ImageSegmentationYOLO("segment", output="text_region", device="cpu", threshold=.7, resolution=64)
    task.context.file_path = tmp_path / "result"
    task._predictor = fake_predictor([mask])
    result, = task.run(image)
    assert result.parents == [image.id] and result.partition == "test"
    assert result.orientation == image.orientation
    np.testing.assert_array_equal(mask_in_view(image, result._mask_image()), mask.astype(np.uint8) * 255)
    assert set(np.unique(np.asarray(PILImage.open(result.mask)))) <= {0, 1}
    assert task._predictor.predict.call_args.kwargs == {
        "conf": .7, "device": "cpu", "retina_masks": True, "verbose": False, "imgsz": 64,
    }


def test_prediction_clips_to_parent_segmentation(sample, tmp_path):
    image = sample.instances[0]
    view = image.pil()
    task = ImageSegmentationYOLO("segment", output="text_region")
    task.context.file_path = tmp_path
    task._predictor = fake_predictor([np.ones((view.height, view.width), bool)])
    result, = task.run(image)
    np.testing.assert_array_equal(result._mask_image(), image._mask_image())


def test_empty_prediction(sample, tmp_path):
    task = ImageSegmentationYOLO("segment")
    task.context.file_path = tmp_path
    task._predictor = SimpleNamespace(predict=Mock(return_value=[SimpleNamespace(masks=None)]))
    assert task.run(sample.image) == []


def test_model_path_and_segmentation_validation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.touch()
    factory = Mock(return_value=SimpleNamespace(task="detect"))
    monkeypatch.setattr(yolo, "get_yolo_class", lambda: factory)
    task = ImageSegmentationYOLO("segment", model="weights.pt")
    task.pipeline = SimpleNamespace(path=tmp_path)
    with pytest.raises(ValueError, match="segmentation model"):
        task._init_model()
    factory.assert_called_once_with(str(checkpoint))


@pytest.mark.parametrize("device,count,expected", [("cuda", 1, 0), ("cuda:2", 2, [2, 3]), ("cpu", 1, "cpu"), ("mps", 1, "mps")])
def test_device_mapping(device, count, expected):
    assert yolo.yolo_device(device, count) == expected


def test_bitmap_dataset_keeps_holes_islands_and_collates(sample, tmp_path):
    pytest.importorskip("ultralytics")
    from collectra.tasks.image_segmentation.yolo_training import BitmapDataset

    directory = export_dataset([sample], tmp_path / "dataset", prompts={"text_region": "text_region"})
    dataset = BitmapDataset(directory / "train", 32)
    item = dataset[0]
    assert item["img"].shape == (3, 32, 32)
    np.testing.assert_array_equal(item["masks"][0].numpy(), np.asarray(sample.instances[0]._mask_image()) != 0)
    batch = dataset.collate_fn([item, dataset[0]])
    assert batch["img"].shape == (2, 3, 32, 32)
    assert batch["batch_idx"].tolist() == [0, 1]
    flipped = BitmapDataset(directory / "train", 32, augment=True, fliplr=1, flipud=1)[0]
    np.testing.assert_array_equal(flipped["masks"].numpy(), item["masks"].numpy()[:, ::-1, ::-1])


def test_train_reuses_coco_data_and_adopts_best(sample, tmp_path, monkeypatch):
    task = ImageSegmentationYOLO("segment", output="text_region", device="cuda:1")
    task.pipeline = SimpleNamespace(path=tmp_path, data={"segment": {}}, save=Mock())
    output = tmp_path / "output"
    export_dataset([sample], output / "dataset", prompts={"text_region": "text_region"})
    task.prepare_training_data = Mock(side_effect=AssertionError("must reuse"))

    def run(command, **kwargs):
        config = json.loads(Path(command[-1]).read_text())
        assert config["params"]["device"] == [1, 2]
        assert config["params"]["batch"] == 6
        assert config["wandb"] and config["project"] == f"{tmp_path.name}-segment"
        (output / "weights").mkdir()
        (output / "weights/best.pt").write_bytes(b"best weights")

    monkeypatch.setattr(yolo.subprocess, "run", run)
    destination = task.train([], output=output, reuse_data=True, max_items=0, batch=3, num_gpus=2, wandb=True)
    task.prepare_training_data.assert_not_called()
    assert destination.read_bytes() == b"best weights"
    assert task.pipeline.data["segment"]["model"] == destination.name


def test_missing_reuse_falls_back_to_export(sample, tmp_path, monkeypatch):
    task = ImageSegmentationYOLO("segment", output="text_region", device="cpu")
    task.pipeline = SimpleNamespace(path=tmp_path, data={"segment": {}}, save=Mock())
    task._samples = Mock(return_value=[sample])
    output = tmp_path / "run"

    def run(command, **kwargs):
        (output / "weights").mkdir()
        (output / "weights/best.pt").write_bytes(b"weights")

    monkeypatch.setattr(yolo.subprocess, "run", run)
    destination = task.train(["source"], output=output, reuse_data=True, exclude="test", max_items=2, keep_log=False)
    task._samples.assert_called_once_with(["source"], max_items=2, exclude="test")
    assert destination.is_file() and not output.exists()


@pytest.mark.parametrize("has_validation", [True, False])
def test_real_cpu_training_and_checkpoint_reload(sample, tmp_path, monkeypatch, has_validation):
    torch = pytest.importorskip("torch")
    ultralytics = pytest.importorskip("ultralytics")
    from collectra.tasks.image_segmentation.yolo_training import launch

    other = deepcopy(sample)
    other.image.id = "other"
    other.image.partition = "validation" if has_validation else ""
    other.instances[0].parents = ["other"]
    other.instances[0].partition = other.image.partition
    dataset = export_dataset([sample, other], tmp_path / "dataset", validation="validation", prompts={"text_region": "text_region"})
    output = tmp_path / "training"
    output.mkdir()
    config = {
        "model": "yolo11n-seg.yaml", "dataset": str(dataset), "threshold": .5,
        "has_validation": has_validation, "wandb": False, "project": "unit-test", "run_name": "run",
        "params": {
            "data": str(output / "config.json"), "project": str(tmp_path), "name": "training",
            "exist_ok": True, "epochs": 1, "batch": 1, "device": "cpu", "imgsz": 64,
            "overlap_mask": False, "mask_ratio": 1, "close_mosaic": 0, "plots": False,
            "workers": 0, "amp": False, "pretrained": False, "verbose": False,
        },
    }
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr("ultralytics.utils.checks.check_pip_update_available", lambda: False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        launch(config_path)
        import csv

        rows = list(csv.DictReader((output / "metrics.csv").open()))
        assert len(rows) == 1 and int(rows[0]["epoch"]) == 1
        assert ("val/mask_iou" in rows[0]) == has_validation
        best = output / "weights/best.pt"
        assert list((output / "weights").glob("*.pt")) == [best]
        checkpoint = torch.load(best, map_location="cpu", weights_only=False)
        assert checkpoint["optimizer"] is None
        assert "scaler" not in checkpoint
        loaded = ultralytics.YOLO(str(best))
        assert loaded.task == "segment" and loaded.names == {0: "text_region"}
        task = ImageSegmentationYOLO("segment", model=best, output="text_region", device="cpu", resolution=64)
        task.context.file_path = tmp_path / "predictions"
        assert isinstance(task.run(sample.image), list)
    finally:
        torch.set_num_threads(threads)
