"""W&B detection curves from RF-DETR's native validation matching data."""

from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path

import numpy as np


def detection_curves(matching, class_names):
    """Use RF-DETR's IoU=0.5 matching and confidence sweep, excluding void classes."""
    from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds

    ids = sorted(matching)
    included = [index for index, cid in enumerate(ids) if matching[cid]["total_gt"] > 0]
    if not included:
        return None
    sweep = sweep_confidence_thresholds(
        [matching[cid] for cid in ids], np.linspace(0, 1, 101), included
    )
    confidence = [float(row["confidence_threshold"]) for row in sweep]
    precision = np.asarray([row["per_class_prec"] for row in sweep])[:, included].T
    recall = np.asarray([row["per_class_rec"] for row in sweep])[:, included].T
    f1 = np.asarray([row["per_class_f1"] for row in sweep])[:, included].T
    recall_grid = np.linspace(0, 1, 101)
    pr = []
    for r, p in zip(recall, precision):
        # Recall repeats while confidence removes false positives. Keep the
        # highest precision at each recall, then form its upper envelope.
        unique = np.unique(r)
        values = np.array([p[r == point].max() for point in unique])
        envelope = np.maximum.accumulate(values[::-1])[::-1]
        pr.append(np.interp(recall_grid, unique, envelope, right=0))
    names = [str(class_names.get(ids[index], ids[index])) for index in included]

    def series(values):
        return np.concatenate([values, np.mean(values, axis=0, keepdims=True)]).tolist()

    return {
        "names": names + ["mean"],
        "iou_threshold": 0.5,
        "confidence": confidence,
        "recall": series(recall),
        "precision": series(precision),
        "f1": series(f1),
        "recall_grid": recall_grid.tolist(),
        "pr_precision": series(np.asarray(pr)),
    }


def log_detection_curves(trainer, curves, output):
    """Log into the trainer's existing W&B run, never create a second run."""
    if curves is None or not trainer.is_global_zero:
        return
    from pytorch_lightning.loggers import WandbLogger
    import wandb

    directory = Path(output) / "curves"
    directory.mkdir(parents=True, exist_ok=True)
    epoch = int(trainer.current_epoch) + 1
    (directory / f"epoch-{epoch:04d}.json").write_text(
        json.dumps({"epoch": epoch, **curves}, indent=2) + "\n"
    )
    plots = {
        "Recall-Confidence": ("confidence", "recall", "Confidence"),
        "Precision-Recall": ("recall_grid", "pr_precision", "Recall"),
        "Precision-Confidence": ("confidence", "precision", "Confidence"),
        "F1-Confidence": ("confidence", "f1", "Confidence"),
    }
    for logger in trainer.loggers:
        if not isinstance(logger, WandbLogger):
            continue
        payload = {
            f"curves/{name}": wandb.plot.line_series(
                xs=curves[x],
                ys=curves[y],
                keys=curves["names"],
                title=f"{name} Curve (IoU 0.5)",
                xname=xlabel,
            )
            for name, (x, y, xlabel) in plots.items()
        }
        payload["trainer/global_step"] = trainer.global_step
        payload["epoch"] = int(trainer.current_epoch)
        logger.experiment.log(payload)


@contextmanager
def wandb_detection_curves(enabled, output, classes):
    """Capture compact matches before the native callback clears its epoch state."""
    if not enabled:
        yield
        return
    from rfdetr.training.callbacks import coco_eval

    callback = coco_eval.COCOEvalCallback
    original = getattr(callback, "_compute_and_log", None)
    if original is None or not callable(
        getattr(coco_eval, "distributed_merge_matching_data", None)
    ):
        raise RuntimeError(
            "RF-DETR validation curves require its Lightning validation matching API. "
            'Update the backend with: pip install --upgrade "rfdetr[train,loggers]"'
        )

    @wraps(original)
    def compute_and_log(self, trainer, pl_module, split, *args, **kwargs):
        # All ranks must enter this collective, including ranks with no local
        # detections. Only rank zero builds plots and writes files afterward.
        matching = None
        if split == "val" and not trainer.sanity_checking:
            matching = coco_eval.distributed_merge_matching_data(self._f1_local)
        result = original(self, trainer, pl_module, split, *args, **kwargs)
        if matching is not None and trainer.is_global_zero:
            names = dict(enumerate(classes))
            names.update(getattr(self, "_cat_id_to_name", {}))
            log_detection_curves(trainer, detection_curves(matching, names), output)
        return result

    callback._compute_and_log = compute_and_log
    try:
        yield
    finally:
        callback._compute_and_log = original
