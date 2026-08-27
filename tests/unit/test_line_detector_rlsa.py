from pathlib import Path

import pytest
from PIL import Image as PillowImage
from PIL import ImageDraw

from collectra import LineDetectorRLSA
from collectra.types.images import Image, ImageCrop


def printed_lines(path: Path, angle: float = 0) -> Image:
    page = PillowImage.new("RGB", (600, 300), "white")
    draw = ImageDraw.Draw(page)
    for y in (45, 125, 205):
        for x, width in ((55, 90), (175, 130), (340, 170)):
            draw.rectangle((x, y, x + width, y + 17), fill="black")
    # A page rule should not be emitted as a text line.
    draw.rectangle((25, 260, 575, 262), fill="black")
    if angle:
        page = page.rotate(
            angle, resample=PillowImage.Resampling.BICUBIC, fillcolor="white"
        )
    page.save(path)
    return Image(name="page", data=path)


def test_detects_printed_lines_and_removes_rules(tmp_path):
    image = printed_lines(tmp_path / "lines.png")
    task = LineDetectorRLSA("detect", output="lines")

    crops = task.run(image)

    assert len(crops) == 3
    assert all(crop.name == "lines" for crop in crops)
    assert [crop.y_center for crop in crops] == sorted(crop.y_center for crop in crops)


def test_deskews_before_finding_line_bands(tmp_path):
    image = printed_lines(tmp_path / "skewed.png", angle=2.5)
    task = LineDetectorRLSA("detect", output="lines", max_skew_degrees=4)

    crops = task.run(image)

    assert len(crops) == 3
    assert all(crop.height_relative < 0.5 for crop in crops)


def test_blank_image_falls_back_to_whole_image(tmp_path):
    path = tmp_path / "blank.png"
    PillowImage.new("RGB", (200, 100), "white").save(path)
    image = Image(name="page", data=path)

    crops = LineDetectorRLSA("detect").run(image)

    assert len(crops) == 1
    assert crops[0].width_relative == pytest.approx(1)
    assert crops[0].height_relative == pytest.approx(1)


def test_nested_crop_coordinates_are_mapped_to_source(tmp_path):
    image = printed_lines(tmp_path / "nested.png")
    region = ImageCrop(
        name="region",
        data=image.data,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.8,
        height_relative=0.8,
    )

    crops = LineDetectorRLSA("detect").run(region)

    assert len(crops) == 3
    assert all(crop.width_relative == pytest.approx(0.8) for crop in crops)
    assert all(0.1 <= crop.y_center <= 0.9 for crop in crops)


def test_accepts_pipeline_configuration_without_model_loading():
    task = LineDetectorRLSA(
        "detect_lines",
        input="apparatus_image",
        output="apparatus_lines",
        deskew=False,
    )

    assert task.input == "apparatus_image"
    assert task.output == "apparatus_lines"
    assert task.deskew is False


def test_rejects_invalid_input_and_parameters():
    with pytest.raises(TypeError, match="image must be Image"):
        LineDetectorRLSA("detect").run("not-an-image")
    with pytest.raises(ValueError, match="padding_fraction"):
        LineDetectorRLSA("detect", padding_fraction=2)
