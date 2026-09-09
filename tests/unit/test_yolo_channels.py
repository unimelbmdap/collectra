"""Dataset-authoritative YOLO input channels, using image headers only."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile
import yaml
from PIL import Image as PILImage

from collectra import Image, ObjectDetectionYOLO
from collectra.types.images import image_channel_count


def label(path):
    path.with_suffix(".txt").write_text("0 0.5 0.5 0.25 0.25\n")
    return path.name


def raster(tmp_path, name, mode):
    path = tmp_path / name
    PILImage.new(mode, (11, 9)).save(path)
    return label(path)


def tiff(tmp_path, name, shape, **kwargs):
    path = tmp_path / name
    tifffile.imwrite(
        path, np.zeros(shape, dtype=np.uint8), photometric="minisblack", **kwargs
    )
    return label(path)


@pytest.mark.parametrize("mode, expected", [("RGB", 3), ("RGBA", 4), ("L", 1)])
def test_raster_config_channels(tmp_path, mode, expected):
    train = raster(tmp_path, "train_image.png", mode)
    val = raster(tmp_path, "val_image.png", mode)
    task = ObjectDetectionYOLO("detector", model="unrelated-checkpoint.pt")
    config = yaml.safe_load(
        task._prepare_yolo_config(tmp_path, ["Palynomorph"], [train], [val]).read_text()
    )
    assert config == {
        "train": "train.txt",
        "val": "val.txt",
        "nc": 1,
        "names": ["Palynomorph"],
        "channels": expected,
    }


@pytest.mark.parametrize(
    "shape, options",
    [
        ((75, 9, 11), {"metadata": {"axes": "CYX"}}),
        ((9, 11, 75), {"metadata": {"axes": "YXC"}}),
        ((75, 9, 11), {"planarconfig": "separate"}),
        ((9, 11, 75), {"planarconfig": "contig"}),
        ((75, 9, 11), {"metadata": None}),
    ],
)
@pytest.mark.parametrize("suffix", [".tif", ".tiff"])
def test_multichannel_tiff_config_and_image_dimensions(
    tmp_path, shape, options, suffix
):
    # 75 is larger than both spatial dimensions, so a smallest-axis guess is wrong.
    name = tiff(tmp_path, "hyperspectral" + suffix, shape, **options)
    task = ObjectDetectionYOLO("detector", model="three-channel-model.pt")
    path = task._prepare_yolo_config(tmp_path, ["Palynomorph"], [name], [])
    assert yaml.safe_load(path.read_text())["channels"] == 75
    image = Image(name="Palynomorph", data=tmp_path / name)
    assert (image.width, image.height) == (11, 9)


def test_tiff_grayscale_and_metadata_only(tmp_path, monkeypatch):
    name = tiff(tmp_path, "gray.tif", (9, 11))
    monkeypatch.setattr(
        tifffile.TiffFile, "asarray", Mock(side_effect=AssertionError("decoded pixels"))
    )
    monkeypatch.setattr(
        tifffile.TiffPage, "asarray", Mock(side_effect=AssertionError("decoded pixels"))
    )
    monkeypatch.setattr(
        tifffile.TiffPageSeries,
        "asarray",
        Mock(side_effect=AssertionError("decoded pixels")),
    )
    assert image_channel_count(tmp_path / name) == 1
    name = tiff(tmp_path, "multi.tif", (75, 9, 11), metadata={"axes": "CYX"})
    assert image_channel_count(tmp_path / name) == 75


@pytest.mark.parametrize("split", ["train", "val"])
def test_inconsistent_channels_fail_before_ultralytics(tmp_path, monkeypatch, split):
    import ultralytics.models

    class FakeYOLO:
        train = Mock()

    monkeypatch.setattr(ultralytics.models, "YOLO", FakeYOLO)
    task = ObjectDetectionYOLO("detector", model=FakeYOLO())
    rgb = raster(tmp_path, "rgb.png", "RGB")
    gray = raster(tmp_path, "gray.png", "L")
    multi = tiff(tmp_path, "multi.tif", (75, 9, 11), metadata={"axes": "CYX"})
    train, val = ([rgb, gray], [multi]) if split == "train" else ([rgb], [gray, multi])
    with pytest.raises(ValueError, match="Inconsistent input channel counts") as error:
        task._train_fold(train, val, ["Palynomorph"], tmp_path, {})
    for detail in [
        "rgb.png: 3 channels",
        "gray.png: 1 channels",
        "multi.tif: 75 channels",
    ]:
        assert detail in str(error.value)
    task.model.train.assert_not_called()
    assert not (tmp_path / "config.yml").exists()


def test_tif_extensions_and_distribution_are_not_double_counted(tmp_path, monkeypatch):
    assert {".tif", ".tiff"} <= set(Image.image_types())
    assert len(Image.image_types()) == len(set(Image.image_types()))
    assert Image.is_image_file(Path("image.tif"))
    assert Image.is_image_file(Path("image.TIFF"))
    train = tiff(tmp_path, "train.tif", (9, 11))
    val = tiff(tmp_path, "val.tiff", (9, 11))
    distribution = Mock()
    monkeypatch.setattr(
        "collectra.tasks.object_detection.yolo.print_distribution_table", distribution
    )
    ObjectDetectionYOLO("detector")._check_distribution(
        tmp_path, {"classes": ["Palynomorph"], "train": [train], "val": [val]}
    )
    distribution.assert_called_once_with(
        "Class Distribution", ["Palynomorph"], {"Palynomorph": 1}, {"Palynomorph": 1}
    )


def test_unreadable_image_error_identifies_file(tmp_path):
    with pytest.raises(ValueError, match="missing.tif"):
        ObjectDetectionYOLO("detector")._prepare_yolo_config(
            tmp_path, ["Palynomorph"], ["missing.tif"], []
        )


def test_ambiguous_volume_axes_are_not_guessed(tmp_path):
    name = tiff(tmp_path, "volume.tif", (75, 9, 11), metadata={"axes": "ZYX"})
    with pytest.raises(ValueError, match="Ambiguous TIFF channel axes.*volume.tif"):
        image_channel_count(tmp_path / name)


def test_prepared_data_is_copied_without_rgb_conversion(tmp_path):
    source = tmp_path / "source.tif"
    tifffile.imwrite(
        source,
        np.zeros((9, 11, 75), dtype=np.uint8),
        planarconfig="contig",
        photometric="minisblack",
    )
    image = Image(name="Palynomorph", data=source, partition="train")
    log = tmp_path / "run"
    log.mkdir()
    task = ObjectDetectionYOLO("detector")
    crop = image.make_crop(0.5, 0.5, 0.5, 0.5, name="Palynomorph")
    crop.partition = "train"
    crop.add_source_parent(image)
    train, val = task._prepare_assets(["Palynomorph"], log, "val", "", crop)
    config = task._prepare_yolo_config(log, ["Palynomorph"], train, val)
    assert yaml.safe_load(config.read_text())["channels"] == 75
    assert (log / train[0]).read_bytes() == source.read_bytes()
