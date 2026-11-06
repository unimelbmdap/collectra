"""Fine-tuning SAM 3 for object detection: boxes only, no masks."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from PIL import Image as PILImage

from collectra import Image, ImageCrop, ImageSegmenterSam3, ObjectDetectorSam3
from collectra.cli import invoke
from collectra.tasks.image_segmentation import sam3
from collectra.tasks.image_segmentation.sam3_metrics import TrainingHooks
from collectra.tasks.image_segmentation.sam3_training import (
    freeze_segmentation_head,
    training_config,
)


@pytest.fixture
def sam_package(tmp_path, monkeypatch):
    package = tmp_path / "sam3"
    template = (
        package / "train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml"
    )
    template.parent.mkdir(parents=True)
    config = {
        "paths": {},
        "scratch": {"collate_fn": {"with_seg_masks": "${scratch.enable_segmentation}"}},
        "roboflow_train": {
            "loss": {"loss_fns_find": [{"_target_": "sam3.train.loss.loss_fns.Boxes"}]},
            "val_transforms": [{"transforms": []}],
        },
        "trainer": {
            "model": {},
            "data": {"train": {"dataset": {}}, "val": {"dataset": {}}},
            "meters": {
                "val": {"roboflow100": {"detection": {"pred_file_evaluators": [{}]}}}
            },
        },
        "launcher": {},
        "submitit": {},
    }
    template.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        sam3, "_sam3", lambda: SimpleNamespace(__file__=str(package / "__init__.py"))
    )
    return package


def coco(tmp_path, validation):
    dataset = tmp_path / "dataset"
    for split in ("train", "valid"):
        (dataset / split).mkdir(parents=True)
        images = [{"id": 0}] if split == "train" or validation else []
        (dataset / split / "_annotations.coco.json").write_text(
            json.dumps({"images": images, "annotations": [], "categories": []})
        )
    return dataset


@pytest.mark.parametrize("validation", [False, True])
def test_box_training_leaves_out_everything_about_masks(
    tmp_path, sam_package, validation
):
    config = training_config(
        coco(tmp_path, validation),
        tmp_path,
        tmp_path / "initial.pt",
        epochs=2,
        batch=1,
        num_gpus=1,
        masks=False,
    )
    assert config["scratch"]["enable_segmentation"] is False
    assert config["trainer"]["data"]["train"]["dataset"]["load_segmentation"] is False
    assert config["roboflow_train"]["loss"]["loss_fns_find"] == [
        {"_target_": "sam3.train.loss.loss_fns.Boxes"}
    ]
    assert config["roboflow_train"]["val_transforms"][0]["transforms"] == []
    assert config["trainer"]["model"]["masks"] is False
    assert config["trainer"]["collectra_selection"] == "bbox_ap"
    if validation:
        meters = config["trainer"]["meters"]["val"]["roboflow100"]
        assert "mask_iou" not in meters
        assert meters["detection"]["pred_file_evaluators"][0]["gt_path"].endswith(
            "valid/_annotations.coco.json"
        )


def test_mask_training_is_unchanged(tmp_path, sam_package):
    config = training_config(
        coco(tmp_path, True),
        tmp_path,
        tmp_path / "initial.pt",
        epochs=2,
        batch=1,
        num_gpus=1,
    )
    assert config["trainer"]["collectra_selection"] == "mask_iou"
    assert config["trainer"]["model"]["masks"] is True
    assert "mask_iou" in config["trainer"]["meters"]["val"]["roboflow100"]


def test_the_segmentation_head_is_frozen_for_box_training():
    torch = pytest.importorskip("torch")
    model = SimpleNamespace(
        segmentation_head=torch.nn.Linear(2, 2), other=torch.nn.Linear(2, 2)
    )
    freeze_segmentation_head(model)
    assert not any(p.requires_grad for p in model.segmentation_head.parameters())
    assert all(p.requires_grad for p in model.other.parameters())
    freeze_segmentation_head(SimpleNamespace())  # a model without the head


def hooked_trainer(tmp_path, metrics, selection="bbox_ap"):
    torch = pytest.importorskip("torch")

    class Upstream:
        def __init__(self, **kwargs):
            self.max_epochs = 1
            self.distributed_rank = 0
            self.model = torch.nn.Linear(1, 1)
            self.epoch = 0

        def _init_model_state(self):
            pass

        def train_epoch(self, loader):
            return {"Losses/train_all_loss": 1.0}

        def val_epoch(self, loader, phase):
            return metrics

    class Trainer(TrainingHooks, Upstream):
        pass

    return Trainer(
        collectra_output=tmp_path,
        collectra_selection=selection,
        data={"val": {"dataset": {}}},
    )


def test_the_best_model_is_chosen_by_box_average_precision(tmp_path):
    trainer = hooked_trainer(
        tmp_path, {"Meters_train/val_roboflow100/detection/coco_eval_bbox_AP": 0.42}
    )
    trainer.train_epoch(None)
    trainer.val_epoch(None, "val")
    best = json.loads((tmp_path / "training/checkpoints/best.json").read_text())
    assert best == {"epoch": 1, "metric": "val/bbox_ap", "value": 0.42}
    (row,) = csv.DictReader((tmp_path / "metrics.csv").open())
    assert float(row["val/bbox_ap"]) == pytest.approx(0.42)


def test_a_validation_without_box_average_precision_is_an_error(tmp_path):
    trainer = hooked_trainer(tmp_path, {"Meters_train/other": 1.0})
    with pytest.raises(ValueError, match="box average precision"):
        trainer.val_epoch(None, "val")


def test_the_selection_metric_is_recorded_in_the_run_config(tmp_path, monkeypatch):
    seen = {}

    class Reporter:
        def __init__(self, *args, config=None, **kwargs):
            seen.update(config)

    monkeypatch.setattr(
        "collectra.tasks.image_segmentation.sam3_metrics.EpochReporter", Reporter
    )
    hooked_trainer(tmp_path, {})
    assert seen["selection_metric"] == "val/bbox_ap"


# --- the detector's own training ----------------------------------------------------


@pytest.fixture
def labelled(tmp_path):
    """Two source pictures: one with an hourglass, one a background."""
    images = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        PILImage.new("RGB", (100, 50)).save(path)
        images.append(Image(name=f"page{index}", id=f"p{index}", data=path))
    crop = images[0].make_crop(0.5, 0.5, 0.2, 0.4, name="hourglass")
    crop.source_parent = images[0]
    return [images[0], images[1], crop]


def trained(task, tmp_path, monkeypatch, images, **kwargs):
    calls = {}

    def fake_trainer(dataset, output, **options):
        calls.update(options, dataset=dataset, output=output)
        for split in ("train", "valid"):
            calls[split] = json.loads(
                (dataset / split / "_annotations.coco.json").read_text()
            )
        best = output / "training/checkpoints/best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"weights")
        return best

    monkeypatch.setattr(task, "_run_trainer", fake_trainer)
    result = task._train(
        *images, log="run", base_folder=tmp_path, project="pipe-detect", **kwargs
    )
    return result, calls


def test_training_writes_a_box_dataset_and_returns_the_best_weights(
    tmp_path, labelled, monkeypatch
):
    task = ObjectDetectorSam3("detect", output=["hourglass"], epochs=7, batch=3)
    result, calls = trained(task, tmp_path, monkeypatch, labelled)
    assert calls["masks"] is False
    assert (calls["epochs"], calls["batch"], calls["num_gpus"]) == (7, 3, 1)
    assert calls["project"] == "pipe-detect"
    (annotation,) = calls["train"]["annotations"]
    assert annotation["category_id"] == 0
    assert len(calls["train"]["images"]) == 2  # the background picture is kept
    assert result.save_dir == tmp_path / "run"
    assert (result.save_dir / "weights/best.pt").read_bytes() == b"weights"


def test_categories_are_named_by_their_prompts(tmp_path, labelled, monkeypatch):
    task = ObjectDetectorSam3(
        "detect", output=["hourglass"], prompts={"hourglass": "an hourglass"}
    )
    _, calls = trained(task, tmp_path, monkeypatch, labelled)
    for split in ("train", "valid"):
        assert [c["name"] for c in calls[split]["categories"]] == ["an hourglass"]


def test_classes_come_from_the_given_classes_else_the_outputs_else_the_crops(
    tmp_path, labelled, monkeypatch
):
    def names(task, **kwargs):
        _, calls = trained(task, tmp_path / "x", monkeypatch, labelled, **kwargs)
        return [c["name"] for c in calls["train"]["categories"]]

    (tmp_path / "x").mkdir()
    assert names(ObjectDetectorSam3("d", output=["a"]), classes=["hourglass"]) == [
        "hourglass"
    ]
    assert names(ObjectDetectorSam3("d", output=["hourglass"])) == ["hourglass"]
    assert names(ObjectDetectorSam3("d")) == ["hourglass"]


def test_training_needs_classes(tmp_path, monkeypatch):
    task = ObjectDetectorSam3("detect")
    with pytest.raises(ValueError, match="No classes"):
        trained(task, tmp_path, monkeypatch, [])


def test_two_outputs_cannot_share_a_prompt_when_training(
    tmp_path, labelled, monkeypatch
):
    task = ObjectDetectorSam3("detect", prompt="label", output=["a", "b"])
    with pytest.raises(ValueError, match="different prompt"):
        trained(task, tmp_path, monkeypatch, labelled)


def test_a_model_given_for_this_run_is_the_one_trained_from(
    tmp_path, labelled, monkeypatch
):
    task = ObjectDetectorSam3("detect", output=["hourglass"], model="old.pt")
    trained(task, tmp_path, monkeypatch, labelled, model="start.pt")
    assert task.model == "start.pt"


def test_training_needs_a_log_directory_and_base_folder(tmp_path):
    task = ObjectDetectorSam3("detect", output=["hourglass"])
    with pytest.raises(ValueError, match="'log'"):
        task._train(base_folder=tmp_path)
    with pytest.raises(ValueError, match="base_folder"):
        task._train(log="run")


def test_the_train_command_uses_the_shared_training_lifecycle(monkeypatch):
    captured = {}

    def fake_command(task, inputs, trainer, **kwargs):
        captured.update(kwargs, inputs=inputs, trainer=trainer)
        return "trained"

    monkeypatch.setattr(
        "collectra.tasks.object_detection.sam3.run_training_command", fake_command
    )
    task = ObjectDetectorSam3("detect", epochs=9, batch=2)
    invoke(
        task,
        ["train", "data", "--validation", "val", "--num-gpus", "2", "--wandb"],
    )
    assert captured["inputs"] == ["data"]
    assert captured["trainer"] == task._train
    assert captured["include_unlabelled"] is True
    assert (captured["epochs"], captured["batch"]) == (9, 2)
    assert captured["validation"] == "val"
    assert (captured["num_gpus"], captured["wandb"]) == (2, True)
    assert captured["model"] == ""


def test_max_items_limits_each_split(tmp_path, labelled, monkeypatch):
    task = ObjectDetectorSam3("detect", output=["hourglass"])
    _, calls = trained(task, tmp_path, monkeypatch, labelled, max_items=1)
    assert len(calls["train"]["images"]) == 1
