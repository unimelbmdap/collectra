import csv
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from collectra.tasks.image_segmentation.sam3_metrics import (
    EpochReporter,
    MaskIoU,
    TrainingHooks,
)


def update_meter(meter, truth, predictions, *, scores=None, presence=None, valid=None):
    import torch

    truth = torch.tensor(truth, dtype=torch.bool)
    predictions = torch.tensor(predictions, dtype=torch.float32)
    outputs = {
        "pred_masks": predictions[None],
        "pred_logits": torch.tensor(scores or [10.0] * len(predictions))[None, :, None],
    }
    if presence is not None:
        outputs["presence_logit_dec"] = torch.tensor([presence])
    target = SimpleNamespace(
        segments=truth,
        num_boxes=torch.tensor([len(truth)]),
        is_valid_segment=torch.tensor(valid or [True] * len(truth)),
    )
    meter.update(find_stages=[outputs], batch=SimpleNamespace(find_targets=[target]))


def test_mask_iou_counts_overlap_false_pixels_and_confidence():
    pytest.importorskip("torch")
    meter = MaskIoU()
    update_meter(
        meter,
        [[[1, 0], [1, 0]]],
        [[[10, 10], [-10, -10]], [[10, 10], [10, 10]]],
        scores=[10.0, -10.0],
    )
    assert meter.compute_synced()["iou"] == pytest.approx(1 / 3)
    assert (meter.intersection, meter.union, meter.queries) == (1, 3, 1)
    meter.reset()
    update_meter(meter, [[[1, 0], [1, 0]]], [[[10, -10], [10, -10]]], presence=-10.0)
    assert meter.compute_synced()["iou"] == 0


def test_mask_iou_weights_pixels_not_query_averages_and_resizes():
    pytest.importorskip("torch")
    meter = MaskIoU()
    update_meter(meter, [[[1, 0], [0, 0]]], [[[10]]])
    update_meter(meter, [[[1]]], [[[10]]])
    assert meter.compute_synced()["iou"] == pytest.approx(2 / 5)
    update_meter(meter, [[[1]]], [[[-10]]], valid=[False])
    assert meter.queries == 2


def test_mask_iou_reduces_global_counts(monkeypatch):
    torch = pytest.importorskip("torch")
    dist = torch.distributed
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(
        dist, "all_reduce", lambda totals: totals.add_(torch.tensor([2, 5, 1]))
    )
    meter = MaskIoU()
    meter.intersection, meter.union, meter.queries = 1, 3, 1
    assert meter.compute_synced()["iou"] == pytest.approx(3 / 8)


def test_mask_iou_uses_packed_target_offsets_for_multiple_queries():
    torch = pytest.importorskip("torch")
    meter = MaskIoU()
    target = SimpleNamespace(
        segments=torch.tensor([[[1, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 1], [0, 0]]]),
        num_boxes=torch.tensor([1, 2]),
        is_valid_segment=torch.ones(3, dtype=torch.bool),
    )
    outputs = {
        "pred_masks": torch.tensor([[[[10.0, -10.0], [-10.0, -10.0]]]]).repeat(
            2, 1, 1, 1
        ),
        "pred_logits": torch.ones(2, 1, 1) * 10,
    }
    meter.update(find_stages=[outputs], batch=SimpleNamespace(find_targets=[target]))
    assert meter.compute_synced()["iou"] == pytest.approx(2 / 3)
    assert meter.queries == 2


def test_mask_iou_counts_false_positives_on_empty_validation_images():
    torch = pytest.importorskip("torch")
    meter = MaskIoU()
    target = SimpleNamespace(
        segments=torch.tensor([]),
        num_boxes=torch.tensor([0]),
        is_valid_segment=torch.tensor([], dtype=torch.bool),
    )
    outputs = {
        "pred_masks": torch.ones(1, 1, 2, 2),
        "pred_logits": torch.ones(1, 1, 1) * 10,
    }
    batch = SimpleNamespace(find_targets=[target], img_batch=torch.zeros(1, 3, 4, 4))
    meter.update(find_stages=[outputs], batch=batch)
    assert meter.compute_synced()["iou"] == 0
    assert meter.union == 16


def test_csv_merges_epochs_and_wandb_uses_the_same_metrics(tmp_path, monkeypatch):
    run = Mock()
    sdk = SimpleNamespace(init=Mock(return_value=run))
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    monkeypatch.setenv("WANDB_PROJECT", "unrelated-project")
    monkeypatch.setenv("WANDB_ENTITY", "my-team")
    reporter = EpochReporter(tmp_path, wandb=True, run_name="run-output", project="pipeline-segment")
    reporter.record(1, "train", {"loss": 3.0})
    reporter.record(1, "val", {"mask_iou": 0.5, "loss": 2.0})
    reporter.record(2, "train", {"loss": 1.0, "extra": 4.0})
    reporter.close()
    rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
    assert len(rows) == 2
    assert rows[0]["train/loss"] == "3.0"
    assert rows[0]["val/mask_iou"] == "0.5"
    assert rows[1]["val/mask_iou"] == ""
    assert rows[1]["train/extra"] == "4.0"
    assert sdk.init.call_args.kwargs["project"] == "pipeline-segment"
    assert sdk.init.call_args.kwargs["name"] == "run-output"
    assert "entity" not in sdk.init.call_args.kwargs
    assert run.log.call_args_list[1].args[0] == {
        "epoch": 1,
        "val/mask_iou": 0.5,
        "val/loss": 2.0,
    }
    run.finish.assert_called_once()


@pytest.mark.parametrize("validation", [True, False])
def test_training_selects_only_best_weights_and_writes_epoch_csv(tmp_path, validation):
    torch = pytest.importorskip("torch")

    class UpstreamTrainer:
        def __init__(self, **kwargs):
            self.distributed_rank = 0
            self.max_epochs = 3
            self.model = torch.nn.Linear(1, 1)
            self.initialized = False
            self.load_checkpoint()

        def _init_model_state(self):
            self.initialized = True

        def train_epoch(self, loader):
            with torch.no_grad():
                self.model.weight.fill_(self.epoch + 1)
            return {"Losses/train_all_loss": [3.0, 1.0, 2.0][self.epoch]}

        def val_epoch(self, loader, phase):
            self.save_checkpoint(self.epoch + 1, ["other_native_best"])
            return {
                "Meters_train/val_roboflow100/mask_iou/iou": [0.1, 0.8, 0.4][self.epoch]
            }

        def run(self):
            for self.epoch in range(3):
                self.train_epoch(None)
                self.save_checkpoint(self.epoch + 1)
                if validation:
                    self.val_epoch(None, "val")

    class Trainer(TrainingHooks, UpstreamTrainer):
        pass

    trainer = Trainer(
        collectra_output=tmp_path, data={"val": {"dataset": {}}} if validation else {}
    )
    trainer.run()
    assert trainer.initialized
    checkpoints = list((tmp_path / "training/checkpoints").glob("*.pt"))
    assert [p.name for p in checkpoints] == ["best.pt"]
    weights = torch.load(checkpoints[0], weights_only=True)
    assert set(weights) == {"weight", "bias"}
    assert weights["weight"].item() == 2
    best = json.loads((checkpoints[0].parent / "best.json").read_text())
    assert best == {
        "epoch": 2,
        "metric": "val/mask_iou" if validation else "train/loss",
        "value": 0.8 if validation else 1.0,
    }
    rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
    assert len(rows) == 3
    assert [row["epoch"] for row in rows] == ["1", "2", "3"]
    assert "val/mask_iou" in rows[0] if validation else "val/mask_iou" not in rows[0]


def test_non_primary_rank_does_not_write_checkpoint(tmp_path):
    pytest.importorskip("torch")
    hooks = object.__new__(TrainingHooks)
    hooks._collectra_best = None
    hooks.distributed_rank = 1
    hooks._collectra_output = tmp_path
    hooks._save_best(0.5, "val/mask_iou", maximize=True)
    assert not list(tmp_path.iterdir())


def test_csv_without_wandb_does_not_import_it(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    reporter = EpochReporter(tmp_path)
    reporter.record(1, "train", {"loss": 2.0})
    reporter.close()
    assert (tmp_path / "metrics.csv").is_file()
