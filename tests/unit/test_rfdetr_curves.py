import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from collectra.tasks.object_detection.rfdetr_curves import (
    detection_curves,
    log_detection_curves,
    wandb_detection_curves,
)


def matching_data(scores=(0.9, 0.8, 0.4), matches=(1, 0, 1), total_gt=2):
    return {
        "scores": np.asarray(scores),
        "matches": np.asarray(matches),
        "ignore": np.zeros(len(scores), dtype=bool),
        "total_gt": total_gt,
    }


def test_curves_use_native_matching_and_exclude_classes_without_ground_truth():
    pytest.importorskip("rfdetr.evaluation.f1_sweep")
    curves = detection_curves(
        {3: matching_data(), 7: matching_data(total_gt=0)}, {3: "page", 7: "empty"}
    )
    assert curves["names"] == ["page", "mean"]
    assert curves["confidence"][50] == 0.5
    assert curves["precision"][0][50] == 0.5
    assert curves["recall"][0][50] == 0.5
    assert curves["f1"][0][50] == 0.5
    assert curves["precision"][0][85] == 1.0
    assert curves["recall"][0][85] == 0.5
    assert curves["precision"][0][0] == pytest.approx(2 / 3)
    assert curves["pr_precision"][0][50] == 1.0
    assert curves["pr_precision"][0][-1] == pytest.approx(2 / 3)
    assert curves["precision"][0] == curves["precision"][1]


def test_curves_retain_missed_targets_and_ignore_crowd_matches():
    pytest.importorskip("rfdetr.evaluation.f1_sweep")
    data = matching_data(scores=(0.9, 0.95), matches=(1, 0), total_gt=2)
    data["ignore"][1] = True
    curves = detection_curves({0: data}, {0: "region"})
    assert curves["precision"][0][0] == 1
    assert curves["recall"][0][0] == 0.5
    assert curves["pr_precision"][0][75] == 0
    assert detection_curves({0: matching_data(total_gt=0)}, {}) is None
    empty = detection_curves({0: matching_data(scores=(), matches=(), total_gt=2)}, {})
    assert not np.asarray(empty["recall"]).any()


def test_four_plots_use_existing_wandb_logger_and_save_curve_data(
    tmp_path, monkeypatch
):
    class WandbLogger:
        experiment = Mock()

    logger = WandbLogger()
    sdk = SimpleNamespace(
        plot=SimpleNamespace(line_series=Mock(side_effect=lambda **kwargs: kwargs))
    )
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    monkeypatch.setitem(
        sys.modules,
        "pytorch_lightning.loggers",
        SimpleNamespace(WandbLogger=WandbLogger),
    )
    curves = {
        "names": ["region", "mean"],
        "confidence": [0, 1],
        "recall_grid": [0, 1],
        "precision": [[1, 0], [1, 0]],
        "recall": [[1, 0], [1, 0]],
        "f1": [[1, 0], [1, 0]],
        "pr_precision": [[1, 1], [1, 1]],
    }
    trainer = SimpleNamespace(
        is_global_zero=True, current_epoch=1, global_step=42, loggers=[object(), logger]
    )
    log_detection_curves(trainer, curves, tmp_path)
    payload = logger.experiment.log.call_args.args[0]
    assert {key for key in payload if key.startswith("curves/")} == {
        "curves/Recall-Confidence",
        "curves/Precision-Recall",
        "curves/Precision-Confidence",
        "curves/F1-Confidence",
    }
    assert payload["trainer/global_step"] == 42
    assert sdk.plot.line_series.call_count == 4
    saved = json.loads((tmp_path / "curves/epoch-0002.json").read_text())
    assert saved == {"epoch": 2, **curves}


@pytest.mark.parametrize(
    "primary,sanity", [(True, False), (False, False), (True, True)]
)
def test_curve_hook_gathers_all_ranks_before_native_reset_and_restores(
    primary, sanity, tmp_path, monkeypatch
):
    module = pytest.importorskip("rfdetr.training.callbacks.coco_eval")
    from collectra.tasks.object_detection import rfdetr_curves

    events = []
    merged = {0: matching_data()}

    def original(callback, trainer, pl_module, split, **kwargs):
        events.append("native")
        callback._f1_local = {}
        return "native result"

    def gather(data):
        events.append("gather")
        return merged

    monkeypatch.setattr(module.COCOEvalCallback, "_compute_and_log", original)
    monkeypatch.setattr(module, "distributed_merge_matching_data", gather)
    build = Mock(return_value={"curves": True})
    log = Mock()
    monkeypatch.setattr(rfdetr_curves, "detection_curves", build)
    monkeypatch.setattr(rfdetr_curves, "log_detection_curves", log)
    callback = object.__new__(module.COCOEvalCallback)
    callback._f1_local = merged if primary else {}
    callback._cat_id_to_name = {0: "actual category"}
    trainer = SimpleNamespace(is_global_zero=primary, sanity_checking=sanity)
    with wandb_detection_curves(True, tmp_path, ["fallback"]):
        assert callback._compute_and_log(trainer, None, "val") == "native result"
    assert events == (["native"] if sanity else ["gather", "native"])
    assert module.COCOEvalCallback._compute_and_log is original
    assert log.call_count == int(primary and not sanity)
    if log.call_count:
        build.assert_called_once_with(merged, {0: "actual category"})


def test_disabled_curves_do_not_import_backends(monkeypatch):
    monkeypatch.setitem(sys.modules, "rfdetr", None)
    monkeypatch.setitem(sys.modules, "wandb", None)
    with wandb_detection_curves(False, "unused", []):
        pass


def test_curve_hook_restores_callback_after_validation_failure(tmp_path, monkeypatch):
    module = pytest.importorskip("rfdetr.training.callbacks.coco_eval")

    def failing(callback, *args, **kwargs):
        raise RuntimeError("validation failed")

    monkeypatch.setattr(module.COCOEvalCallback, "_compute_and_log", failing)
    monkeypatch.setattr(module, "distributed_merge_matching_data", lambda data: data)
    callback = object.__new__(module.COCOEvalCallback)
    callback._f1_local = {}
    trainer = SimpleNamespace(is_global_zero=True, sanity_checking=False)
    with pytest.raises(RuntimeError, match="validation failed"):
        with wandb_detection_curves(True, tmp_path, []):
            callback._compute_and_log(trainer, None, "val")
    assert module.COCOEvalCallback._compute_and_log is failing
