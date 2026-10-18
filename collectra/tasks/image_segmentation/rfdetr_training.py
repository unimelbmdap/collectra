"""Optional RF-DETR training integration; no backend imports during discovery."""

from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path


from .mask_metrics import mask_overlap


def make_callback(output, threshold, has_validation):
    """Build the optional Lightning callback after training dependencies load."""
    import torch
    from pytorch_lightning.callbacks import Callback

    class SegmentationMetrics(Callback):
        def __init__(self):
            self.best = -math.inf
            self.rows = {}

        def on_fit_start(self, trainer, pl_module):
            if trainer.is_global_zero:
                for name in ("best.pt", "best.json"):
                    (Path(output) / name).unlink(missing_ok=True)

        def on_validation_epoch_start(self, trainer, pl_module):
            self.overlap = torch.zeros(2, dtype=torch.int64, device=pl_module.device)

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if trainer.sanity_checking:
                return
            for prediction, target in zip(outputs["results"], outputs["targets"]):
                self.overlap += mask_overlap(prediction, target, threshold)

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(self.overlap)
            score = self.overlap[0] / self.overlap[1].clamp_min(1)
            pl_module.log("val/mask_iou", score.float(), prog_bar=True, sync_dist=False)

        def _report(self, trainer):
            if not trainer.is_global_zero or trainer.sanity_checking:
                return
            row = self.rows.setdefault(trainer.current_epoch + 1, {"epoch": trainer.current_epoch + 1})
            for key, value in trainer.callback_metrics.items():
                if hasattr(value, "numel") and value.numel() != 1:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                row[key] = value
            fields = ["epoch"] + sorted({key for row in self.rows.values() for key in row if key != "epoch"})
            with (Path(output) / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.rows.values())

        def _save(self, trainer, pl_module, metric, score):
            if not trainer.is_global_zero or not math.isfinite(score) or score <= self.best:
                return
            self.best = score
            args = pl_module.train_config.model_dump()
            args["class_names"] = list(trainer.datamodule.class_names)
            config = pl_module.model_config.model_dump()
            checkpoint = {
                "model": pl_module.model.state_dict(), "args": args,
                "model_config": config, "model_name": config["model_name"],
                "epoch": trainer.current_epoch,
            }
            temporary = Path(output) / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(Path(output) / "best.pt")
            (Path(output) / "best.json").write_text(json.dumps({
                "epoch": trainer.current_epoch + 1, "metric": metric,
                "value": -score if metric == "train/loss" else score,
            }), encoding="utf-8")

        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            self._report(trainer)
            score = trainer.callback_metrics.get("val/mask_iou")
            if score is not None:
                self._save(trainer, pl_module, "val/mask_iou", float(score))

        def on_train_epoch_end(self, trainer, pl_module):
            self._report(trainer)
            if not has_validation:
                score = trainer.callback_metrics.get("train/loss")
                if score is not None:
                    self._save(trainer, pl_module, "train/loss", -float(score))

    return SegmentationMetrics()


@contextmanager
def training_hooks(output, threshold, has_validation):
    """Replace native optimizer snapshots with one best, reloadable weights file.

    RF-DETR's train() constructs its own callbacks. Scope the extension to this
    training process, leaving its COCO evaluation and W&B/CSV loggers intact.
    """
    import rfdetr.training.trainer as native
    from pytorch_lightning.callbacks import ModelCheckpoint

    original = native._append_training_callbacks

    def append(callbacks, loggers, *args, **kwargs):
        original(callbacks, loggers, *args, **kwargs)
        callbacks[:] = [cb for cb in callbacks if not isinstance(cb, ModelCheckpoint)]
        callbacks.append(make_callback(output, threshold, has_validation))

    # Also disable Lightning's implicit default checkpoint when none remains.
    import rfdetr.training as training
    original_builder = training.build_trainer

    def build(*args, **kwargs):
        kwargs["enable_checkpointing"] = False
        if not has_validation:
            kwargs["limit_val_batches"] = 0
        return original_builder(*args, **kwargs)

    native._append_training_callbacks = append
    training.build_trainer = build
    try:
        yield
    finally:
        native._append_training_callbacks = original
        training.build_trainer = original_builder


def launch(config_path):
    from .rfdetr import ImageSegmentationRFDETR

    config = json.loads(Path(config_path).read_text())
    task = ImageSegmentationRFDETR(
        "train", model=config["model"], device=config["device"], resolution=config["resolution"],
    )
    with training_hooks(config["params"]["output_dir"], config["threshold"], config["has_validation"]):
        task._init_model()
        task._predictor.train(**config["params"])


if __name__ == "__main__":
    import sys

    launch(sys.argv[1])
