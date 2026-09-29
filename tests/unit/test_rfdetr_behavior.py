"""Behavioral unit tests for ObjectDetectionRFDETR.

These tests pin down three behaviors that the production code currently
violates. They are written to fail against the current implementation in
``collectra/tasks/machine_learning/rfdetr.py`` and pass once the agent
implementing the fixes lands their changes.

Behaviors covered:
1. ``_train_fold`` must instantiate ``RFDETRBase`` with
   ``pretrain_weights=<original_model_path>`` so a user-supplied
   checkpoint is honored during training, and ``.train(...)`` must be
   invoked on *that* instance (not a bare default).
2. ``train()`` must validate the configured model up-front (via
   ``_init_model()``) and raise ``ValueError`` for nonexistent paths or
   paths that ``_load`` would reject before any dataset prep happens.
3. ``_load`` must emit a warning on the module logger when a real
   checkpoint is loaded but no sibling ``classes.json`` is present, and
   stay silent when ``classes.json`` is present.
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The local .venv currently has a corrupt opencv-python (the metadata is
# present but the cv2/ package is missing), so `import ultralytics` -
# which the production module under test transitively requires via
# `from ultralytics.utils import ThreadingLocked` - fails at import time.
# Drop a no-op stub if and only if cv2 cannot be imported, so these tests
# can still run as a pure unit-test eval. Once the env is fixed this
# branch becomes a harmless no-op.
try:  # pragma: no cover - environment guard
    import cv2  # noqa: F401
except Exception:  # pragma: no cover - environment guard
    sys.modules["cv2"] = MagicMock()

from collectra.tasks.object_detection import rfdetr as rfdetr_module  # noqa: E402
from collectra.tasks.object_detection.rfdetr import ObjectDetectionRFDETR  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRFDETR:
    """Common parent for the fake model variants."""


class _FakeRFDETRBase(_FakeRFDETR):
    """A real (non-mock) class we patch in for ``RFDETRBase``.

    Using a real class - rather than a ``MagicMock`` instance - keeps
    ``isinstance(x, RFDETRBase)`` calls inside the production code from
    blowing up with ``TypeError: isinstance() arg 2 must be a type``.

    Each instance records its own constructor kwargs and exposes a
    ``train`` MagicMock so tests can assert on training calls.
    """

    instances: list["_FakeRFDETRBase"] = []

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.train = MagicMock(name="RFDETRBase.train")
        self.predict = MagicMock(name="RFDETRBase.predict")
        type(self).instances.append(self)


@pytest.fixture
def fake_checkpoint(tmp_path: Path) -> Path:
    """Create a real-on-disk checkpoint file (contents irrelevant; the
    underlying RFDETRBase loader is mocked everywhere it's referenced)."""
    ckpt = tmp_path / "weights" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"not-a-real-checkpoint")
    return ckpt


@pytest.fixture
def patched_rfdetr_base():
    """Patch the RFDETRBase symbol used inside rfdetr.py with a real
    class. Yields the patched class so tests can introspect instances and
    constructor calls via ``cls.instances``.
    """
    _FakeRFDETRBase.instances = []
    with patch(
        "collectra.tasks.object_detection.rfdetr.RFDETRBase",
        new=_FakeRFDETRBase,
    ) as patched, patch.object(rfdetr_module, "RFDETR", _FakeRFDETR):
        yield patched


def test_init_model_accepts_other_rfdetr_variants(patched_rfdetr_base):
    class OtherRFDETR(_FakeRFDETR):
        pass

    model = OtherRFDETR()
    task = ObjectDetectionRFDETR(name="rfdetr-other", model=model)

    task._init_model()
    task._init_model()

    assert task.model is model


@pytest.mark.parametrize(
    "name, class_name",
    [
        ("nano", "RFDETRNano"),
        ("small", "RFDETRSmall"),
        ("medium", "RFDETRMedium"),
        ("large", "RFDETRLarge"),
        ("base", "RFDETRBase"),
        ("default", "RFDETRBase"),
        ("RFDETRSmall", "RFDETRSmall"),
        ("xlarge", "RFDETRXLarge"),
        ("2xlarge", "RFDETR2XLarge"),
    ],
)
def test_load_named_variant(name, class_name, monkeypatch):
    model = _FakeRFDETR()
    constructor = MagicMock(return_value=model)
    monkeypatch.setattr(
        rfdetr_module,
        "_backend_class",
        lambda name: _FakeRFDETR if name == "RFDETR" else (
            constructor if name == class_name else pytest.fail(f"Unexpected model: {name}")
        ),
    )
    task = ObjectDetectionRFDETR(name="detector", model=name)
    task.original_model_path = Path("previous.pt")
    monkeypatch.setattr(task, "_select_device", lambda: "cpu")

    task._init_model()

    assert task.model is model
    assert task.original_model_path is None
    constructor.assert_called_once_with()


@pytest.mark.parametrize("class_name", ["RFDETRNano", "RFDETRSmall"])
def test_load_checkpoint_tries_smaller_variants(class_name, monkeypatch, fake_checkpoint):
    model = _FakeRFDETR()
    constructor = MagicMock(return_value=model)
    incompatible = MagicMock(side_effect=ValueError("incompatible checkpoint"))
    monkeypatch.setattr(
        rfdetr_module,
        "_backend_class",
        lambda name: constructor if name == class_name else incompatible,
    )

    assert rfdetr_module.load_rfdetr_model(fake_checkpoint) is model
    constructor.assert_called_once_with(pretrain_weights=str(fake_checkpoint))


# ---------------------------------------------------------------------------
# Behavior 1: training honors the local checkpoint path
# ---------------------------------------------------------------------------


def test_train_fold_uses_pretrain_weights_from_original_model_path(
    tmp_path: Path,
    fake_checkpoint: Path,
    patched_rfdetr_base,
):
    """When the task is configured with a local checkpoint, ``_train_fold``
    must instantiate ``RFDETRBase(pretrain_weights=<that path>)`` for the
    model that gets trained - not a bare ``RFDETRBase()``.
    """
    task = ObjectDetectionRFDETR(name="rfdetr-train", model=fake_checkpoint)
    # Mirror the real call path: train() runs _init_model() before _train_fold,
    # so self.model gets constructed via _load with pretrain_weights wired.
    task._init_model()

    log = tmp_path / "run"
    log.mkdir(parents=True, exist_ok=True)

    classes = ["a", "b"]

    # Skip the heavy COCO writer (PIL / shutil) - none of that is under
    # test here. Also skip _reload so it doesn't re-construct the model
    # with pretrain_weights itself; that would mask the real bug.
    with patch.object(
        ObjectDetectionRFDETR, "_write_coco_dataset", return_value=log / "dataset"
    ), patch.object(ObjectDetectionRFDETR, "_reload"):
        task._train_fold(
            train_samples=[],
            val_samples=[],
            classes=classes,
            log=log,
            kwargs={"base_folder": tmp_path, "wandb": False},
        )

    # Find the instance whose .train(...) was called - that is the model
    # that was actually trained.
    trained_instances = [
        inst for inst in _FakeRFDETRBase.instances if inst.train.called
    ]
    assert trained_instances, (
        "Expected RFDETRBase(...).train(...) to be called once during "
        "_train_fold, but no instance had .train() invoked. "
        f"Instances seen: {len(_FakeRFDETRBase.instances)}"
    )
    assert len(trained_instances) == 1, (
        "Expected exactly one RFDETRBase instance to be trained per fold, "
        f"got {len(trained_instances)}."
    )

    trained = trained_instances[0]
    assert trained.init_kwargs.get("pretrain_weights") == str(fake_checkpoint), (
        "The RFDETRBase instance that was trained must have been constructed "
        f"with pretrain_weights={str(fake_checkpoint)!r}, but got "
        f"init_kwargs={trained.init_kwargs!r}."
    )


# ---------------------------------------------------------------------------
# Behavior 2: train() validates the model up-front
# ---------------------------------------------------------------------------


def test_train_raises_for_nonexistent_model_path(
    tmp_path: Path,
    patched_rfdetr_base,
):
    """A bogus checkpoint path must cause ``train()`` to raise before any
    dataset prep runs. Today, ``train()`` skips ``_init_model()`` entirely
    so bad paths are silently accepted.
    """
    bogus = tmp_path / "does_not_exist.pt"
    assert not bogus.exists()

    task = ObjectDetectionRFDETR(name="rfdetr-train", model=bogus)

    # Spy on the dataset writer to confirm we error out *before* it runs.
    with patch.object(ObjectDetectionRFDETR, "_write_coco_dataset") as write_mock:
        with pytest.raises(ValueError):
            task._train(
                log="run",
                base_folder=tmp_path,
                classes=["a"],
            )
        assert (
            write_mock.call_count == 0
        ), "Expected train() to raise before dataset prep ran."


def test_train_raises_when_load_returns_non_rfdetrbase(
    tmp_path: Path,
    fake_checkpoint: Path,
    patched_rfdetr_base,
):
    """If ``_load`` resolves to something that is not an ``RFDETRBase``
    instance, ``train()`` must raise via ``_init_model``'s isinstance
    check before dataset prep happens.

    We simulate this by patching ``_load`` to leave ``self.model`` as a
    non-RFDETRBase value.
    """
    task = ObjectDetectionRFDETR(name="rfdetr-train", model=fake_checkpoint)

    def _bad_load(self):
        # Pretend loading "succeeded" but produced the wrong type.
        self.model = object()

    with patch.object(
        ObjectDetectionRFDETR, "_load", autospec=True, side_effect=_bad_load
    ):
        with patch.object(ObjectDetectionRFDETR, "_write_coco_dataset") as write_mock:
            with pytest.raises(ValueError):
                task._train(
                    log="run",
                    base_folder=tmp_path,
                    classes=["a"],
                )
            assert (
                write_mock.call_count == 0
            ), "Expected train() to raise before dataset prep ran."


# ---------------------------------------------------------------------------
# Behavior 3: _load warns when classes.json is missing alongside the checkpoint
# ---------------------------------------------------------------------------


def test_load_warns_when_classes_json_missing(
    tmp_path: Path,
    fake_checkpoint: Path,
    patched_rfdetr_base,
    caplog,
):
    """Loading a checkpoint whose parent dir has no ``classes.json`` must
    emit a warning on the module logger.
    """
    # Sanity: make sure no classes.json sits next to the checkpoint.
    sibling = fake_checkpoint.parent / "classes.json"
    assert not sibling.exists()

    task = ObjectDetectionRFDETR(name="rfdetr-load", model=fake_checkpoint)

    with caplog.at_level(logging.WARNING, logger=rfdetr_module.logger.name):
        task._load()

    warnings = [
        rec
        for rec in caplog.records
        if rec.name == rfdetr_module.logger.name and rec.levelno >= logging.WARNING
    ]
    assert warnings, (
        "Expected a warning to be logged on "
        f"{rfdetr_module.logger.name!r} when classes.json is missing, "
        "but got records: "
        f"{[(r.name, r.levelname, r.message) for r in caplog.records]!r}"
    )


def test_load_does_not_warn_when_classes_json_present(
    tmp_path: Path,
    fake_checkpoint: Path,
    patched_rfdetr_base,
    caplog,
):
    """When ``classes.json`` IS present next to the checkpoint, ``_load``
    should not emit the missing-classes warning."""
    classes_file = fake_checkpoint.parent / "classes.json"
    classes_file.write_text(json.dumps({"classes": ["a", "b"]}))

    task = ObjectDetectionRFDETR(name="rfdetr-load", model=fake_checkpoint)

    with caplog.at_level(logging.WARNING, logger=rfdetr_module.logger.name):
        task._load()

    missing_warnings = [
        rec
        for rec in caplog.records
        if rec.name == rfdetr_module.logger.name
        and rec.levelno >= logging.WARNING
        and "classes.json" in rec.getMessage().lower()
    ]
    assert not missing_warnings, (
        "Did not expect a missing-classes.json warning when the file is present. "
        f"Got: {[(r.name, r.levelname, r.message) for r in caplog.records]!r}"
    )
