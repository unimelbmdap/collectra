"""Hermetic unit tests for ``_prepare_assets`` on DETR-family detection tasks.

These tests verify that ``ObjectDetectionRFDETR`` and ``ObjectDetectionDETR``
share YOLO's contract for turning a stream of ``Image``/``ImageCrop`` objects
into training samples. The key behaviour under test is that images without
in-class annotations are retained as empty-box *background* samples rather
than being silently dropped.

Each test is parameterised across both task classes so any drift between
the two implementations is caught. The tests construct real (tiny) PNG
files via PIL in ``tmp_path`` because ``Image.__post_init__`` eagerly
loads the file to read its dimensions.

Tests never call ``_init_model`` / ``train`` / ``run`` — they only
exercise the pure data-transformation method ``_prepare_assets``.
"""

from pathlib import Path

import pytest
from PIL import Image as PILImage

from collectra import Image, ImageCrop
from collectra.tasks.object_detection.detr import ObjectDetectionDETR
from collectra.tasks.object_detection.rfdetr import ObjectDetectionRFDETR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK_CLASSES = [ObjectDetectionRFDETR, ObjectDetectionDETR]
TASK_IDS = ["rfdetr", "detr"]


def _make_png(tmp_path: Path, name: str, size: tuple[int, int]) -> Path:
    """Create a tiny PNG on disk and return its path."""
    path = tmp_path / name
    PILImage.new("RGB", size, color=(255, 255, 255)).save(path)
    return path


def _make_parent(
    tmp_path: Path,
    name: str = "sheet.png",
    size: tuple[int, int] = (1000, 800),
    partition: str = "",
) -> Image:
    """Build a bare parent ``Image`` backed by a real PNG on disk."""
    path = _make_png(tmp_path, name, size)
    return Image(name="specimen_sheet", data=path, partition=partition)


def _make_crop(
    tmp_path: Path,
    crop_name: str,
    *,
    parent: Image | None = None,
    x_center: float = 0.5,
    y_center: float = 0.5,
    width_relative: float = 0.2,
    height_relative: float = 0.2,
    partition: str = "",
    data_name: str | None = None,
) -> ImageCrop:
    """Build an ``ImageCrop`` (with its own on-disk backing file) and
    optionally attach it to ``parent`` via ``add_source_parent``."""
    data_name = data_name or f"{crop_name}_crop.png"
    crop_path = _make_png(tmp_path, data_name, size=(100, 100))
    crop = ImageCrop(
        name=crop_name,
        data=crop_path,
        x_center=x_center,
        y_center=y_center,
        width_relative=width_relative,
        height_relative=height_relative,
        partition=partition,
    )
    if parent is not None:
        crop.add_source_parent(parent)
    return crop


def _all_samples(train: list[dict], val: list[dict]) -> list[dict]:
    return list(train) + list(val)


# ---------------------------------------------------------------------------
# Instantiation sanity check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_instantiation_does_not_load_model(task_cls):
    """Constructing the task with only a name must not load a model."""
    task = task_cls(name="test")
    assert task.name == "test"
    # No model should have been loaded eagerly.
    assert task.model is None


# ---------------------------------------------------------------------------
# Behaviour 1: bare Image becomes a background training sample.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_bare_image_produces_empty_background_sample(task_cls, tmp_path):
    task = task_cls(name="test")
    parent = _make_parent(tmp_path)

    train, val = task._prepare_assets(["label"], "", "", parent)

    assert len(val) == 0
    assert len(train) == 1
    sample = train[0]
    assert sample["image_path"] == parent.get_path()
    assert sample["boxes"] == []
    assert sample["labels"] == []
    assert sample["partition"] == ""


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_bare_image_routes_to_val_when_partition_matches(task_cls, tmp_path):
    task = task_cls(name="test")
    parent = _make_parent(tmp_path, partition="val")

    train, val = task._prepare_assets(["label"], "val", "", parent)

    assert len(train) == 0
    assert len(val) == 1
    assert val[0]["boxes"] == []
    assert val[0]["labels"] == []
    assert val[0]["partition"] == "val"


# ---------------------------------------------------------------------------
# Behaviour 2: in-class crop with parent contributes a single annotation on
# the parent's sample, with absolute-pixel box math.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_in_class_crop_with_parent_adds_single_annotation(task_cls, tmp_path):
    task = task_cls(name="test")
    parent = _make_parent(tmp_path, size=(1000, 800))

    # Crop centred at (0.5, 0.5) with rel size 0.2 x 0.4:
    #   width 0.2 * 1000 = 200, so x1=400, x2=600
    #   height 0.4 * 800 = 320, so y1=240, y2=560
    crop = _make_crop(
        tmp_path,
        "label",
        parent=parent,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.2,
        height_relative=0.4,
    )

    train, val = task._prepare_assets(["label"], "", "", crop)

    assert len(val) == 0
    assert len(train) == 1
    sample = train[0]
    assert sample["image_path"] == parent.get_path()
    assert sample["labels"] == [0]
    assert len(sample["boxes"]) == 1
    box = sample["boxes"][0]
    assert len(box) == 4
    x1, y1, x2, y2 = box
    assert x1 == pytest.approx(400.0)
    assert y1 == pytest.approx(240.0)
    assert x2 == pytest.approx(600.0)
    assert y2 == pytest.approx(560.0)


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_in_class_crop_label_is_classes_index(task_cls, tmp_path):
    task = task_cls(name="test")
    parent = _make_parent(tmp_path)
    crop = _make_crop(
        tmp_path,
        "barcode",
        parent=parent,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.2,
        height_relative=0.2,
    )

    classes = ["label", "barcode", "qr"]
    train, _ = task._prepare_assets(classes, "", "", crop)

    assert len(train) == 1
    assert train[0]["labels"] == [1]


# ---------------------------------------------------------------------------
# Behaviour 3: multiple in-class crops on same parent -> one grouped sample.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_multiple_in_class_crops_group_on_single_parent_sample(task_cls, tmp_path):
    task = task_cls(name="test")
    parent = _make_parent(tmp_path, size=(1000, 800))

    crop_a = _make_crop(
        tmp_path,
        "label",
        parent=parent,
        x_center=0.25,
        y_center=0.5,
        width_relative=0.2,
        height_relative=0.2,
        data_name="crop_a.png",
    )
    crop_b = _make_crop(
        tmp_path,
        "label",
        parent=parent,
        x_center=0.75,
        y_center=0.5,
        width_relative=0.2,
        height_relative=0.2,
        data_name="crop_b.png",
    )

    train, val = task._prepare_assets(["label"], "", "", crop_a, crop_b)

    assert len(val) == 0
    assert len(train) == 1
    sample = train[0]
    assert sample["image_path"] == parent.get_path()
    assert len(sample["boxes"]) == 2
    assert len(sample["labels"]) == 2
    assert sample["labels"] == [0, 0]


# ---------------------------------------------------------------------------
# Behaviour 4: out-of-class crop keeps parent as background sample.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_out_of_class_crop_retains_parent_as_background(task_cls, tmp_path):
    task = task_cls(name="test")
    parent = _make_parent(tmp_path)
    crop = _make_crop(
        tmp_path,
        "barcode",
        parent=parent,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.2,
        height_relative=0.2,
    )

    train, val = task._prepare_assets(["label"], "", "", crop)

    assert len(val) == 0
    assert len(train) == 1
    sample = train[0]
    assert sample["image_path"] == parent.get_path()
    assert sample["boxes"] == []
    assert sample["labels"] == []


# ---------------------------------------------------------------------------
# Behaviour 5: mixed in-class and out-of-class crops on same parent.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_mixed_in_class_and_out_of_class_same_parent(task_cls, tmp_path):
    task = task_cls(name="test")
    parent = _make_parent(tmp_path)

    good = _make_crop(
        tmp_path,
        "label",
        parent=parent,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.2,
        height_relative=0.2,
        data_name="good_crop.png",
    )
    bad = _make_crop(
        tmp_path,
        "barcode",
        parent=parent,
        x_center=0.3,
        y_center=0.3,
        width_relative=0.2,
        height_relative=0.2,
        data_name="bad_crop.png",
    )

    train, val = task._prepare_assets(["label"], "", "", good, bad)

    assert len(val) == 0
    # Only one sample, keyed on parent, containing only the in-class annotation.
    assert len(train) == 1
    sample = train[0]
    assert sample["image_path"] == parent.get_path()
    assert len(sample["boxes"]) == 1
    assert sample["labels"] == [0]


# ---------------------------------------------------------------------------
# Behaviour 6: validation_flag routing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_validation_flag_routes_matching_crop_to_val(task_cls, tmp_path):
    task = task_cls(name="test")
    parent = _make_parent(tmp_path, partition="val")
    crop = _make_crop(
        tmp_path,
        "label",
        parent=parent,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.2,
        height_relative=0.2,
        partition="val",
    )

    train, val = task._prepare_assets(["label"], "val", "", crop)

    assert len(train) == 0
    assert len(val) == 1
    assert val[0]["partition"] == "val"
    assert len(val[0]["boxes"]) == 1


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_empty_validation_flag_sends_everything_to_train(task_cls, tmp_path):
    task = task_cls(name="test")
    # A bare image with a non-empty partition; since validation_flag="" nothing
    # should route to val.
    parent = _make_parent(tmp_path, partition="val")

    train, val = task._prepare_assets(["label"], "", "", parent)

    assert len(val) == 0
    assert len(train) == 1
    assert train[0]["partition"] == "val"


# ---------------------------------------------------------------------------
# Behaviour 7: exclude_flag routing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_exclude_flag_drops_from_both_splits(task_cls, tmp_path):
    task = task_cls(name="test")
    parent_keep = _make_parent(tmp_path, name="keep.png")
    parent_drop = _make_parent(
        tmp_path,
        name="drop.png",
        partition="skip",
    )

    train, val = task._prepare_assets(
        ["label"],
        "",
        "skip",
        parent_keep,
        parent_drop,
    )

    assert len(val) == 0
    assert len(train) == 1
    assert train[0]["image_path"] == parent_keep.get_path()


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_empty_exclude_flag_excludes_nothing(task_cls, tmp_path):
    task = task_cls(name="test")
    # Partition is a non-empty string but exclude_flag=="" so it should not be
    # excluded.
    parent = _make_parent(tmp_path, partition="anything")

    train, val = task._prepare_assets(["label"], "", "", parent)

    assert len(val) == 0
    assert len(train) == 1
    assert train[0]["partition"] == "anything"


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_exclude_flag_drops_crop_even_if_in_class(task_cls, tmp_path):
    task = task_cls(name="test")
    parent = _make_parent(tmp_path, partition="skip")
    crop = _make_crop(
        tmp_path,
        "label",
        parent=parent,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.2,
        height_relative=0.2,
        partition="skip",
    )

    train, val = task._prepare_assets(["label"], "", "skip", crop)

    assert train == []
    assert val == []


# ---------------------------------------------------------------------------
# Behaviour 8: zero-box samples are kept, not dropped.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_three_bare_images_all_kept_as_background(task_cls, tmp_path):
    task = task_cls(name="test")
    parents = [_make_parent(tmp_path, name=f"sheet_{i}.png") for i in range(3)]

    train, val = task._prepare_assets(["label"], "", "", *parents)

    assert len(val) == 0
    assert len(train) == 3
    for sample in train:
        assert sample["boxes"] == []
        assert sample["labels"] == []
    seen_paths = {s["image_path"] for s in train}
    assert seen_paths == {p.get_path() for p in parents}


# ---------------------------------------------------------------------------
# Behaviour 9: degenerate box skip does not evict parent.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_degenerate_box_is_skipped_but_parent_kept(task_cls, tmp_path):
    task = task_cls(name="test")
    parent = _make_parent(tmp_path, size=(1000, 800))

    # width_relative=0 makes x2 == x1 which the impl should skip, while the
    # parent should still appear as an empty background sample.
    crop = _make_crop(
        tmp_path,
        "label",
        parent=parent,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.0,
        height_relative=0.2,
    )

    train, val = task._prepare_assets(["label"], "", "", crop)

    assert len(val) == 0
    assert len(train) == 1
    sample = train[0]
    assert sample["image_path"] == parent.get_path()
    assert sample["boxes"] == []
    assert sample["labels"] == []


# ---------------------------------------------------------------------------
# Behaviour: mixed bare image + in-class crop on a different parent.
# Sanity check to ensure neither interferes with the other.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", TASK_CLASSES, ids=TASK_IDS)
def test_bare_image_and_separate_crop_produce_two_samples(task_cls, tmp_path):
    task = task_cls(name="test")
    parent_bg = _make_parent(tmp_path, name="bg.png")
    parent_fg = _make_parent(tmp_path, name="fg.png")
    crop = _make_crop(
        tmp_path,
        "label",
        parent=parent_fg,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.2,
        height_relative=0.2,
    )

    train, val = task._prepare_assets(
        ["label"],
        "",
        "",
        parent_bg,
        crop,
    )

    assert len(val) == 0
    assert len(train) == 2
    paths = {s["image_path"]: s for s in train}
    assert parent_bg.get_path() in paths
    assert parent_fg.get_path() in paths
    assert paths[parent_bg.get_path()]["boxes"] == []
    assert paths[parent_bg.get_path()]["labels"] == []
    assert len(paths[parent_fg.get_path()]["boxes"]) == 1
    assert paths[parent_fg.get_path()]["labels"] == [0]
