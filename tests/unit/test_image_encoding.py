"""Images are encoded for language models in a format the models accept.

Model APIs such as Anthropic's take only JPEG, PNG, GIF and WebP, so an image in
any other format, such as a TIFF scan or a crop of one, is converted. It is sent
as a high-quality JPEG, which keeps large crops of big scans under the APIs' size
limits, or as PNG when it has transparency, which JPEG cannot keep.
"""

import base64
import io

import pytest
from PIL import Image as PILImage

from collectra import Image

ACCEPTED = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def decoded(image: Image) -> PILImage.Image:
    return PILImage.open(io.BytesIO(base64.b64decode(image.get_encoding())))


@pytest.mark.parametrize(
    "suffix, mime, pil_format",
    [
        (".jpg", "image/jpeg", "JPEG"),
        (".png", "image/png", "PNG"),
        (".gif", "image/gif", "GIF"),
        (".webp", "image/webp", "WEBP"),
    ],
)
def test_accepted_formats_are_sent_as_they_are(tmp_path, suffix, mime, pil_format):
    path = tmp_path / f"page{suffix}"
    PILImage.new("RGB", (40, 30), "white").save(path)
    image = Image(name="page", data=path)
    assert image.mime() == mime
    assert decoded(image).format == pil_format


@pytest.mark.parametrize("suffix", [".tif", ".tiff", ".bmp"])
@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_other_formats_are_sent_as_high_quality_jpeg(tmp_path, suffix, mode):
    path = tmp_path / f"page{suffix}"
    PILImage.new(mode, (40, 30), "white").save(path)
    image = Image(name="page", data=path)
    assert image.mime() == "image/jpeg"
    sent = decoded(image)
    assert sent.format == "JPEG" and sent.size == (40, 30)
    # Quality 95 quantises at most by 12; the default quality (75) by up to 61.
    assert max(sent.quantization[0]) <= 15


@pytest.mark.parametrize("mode", ["RGBA", "LA", "I;16"])
def test_images_jpeg_cannot_hold_are_sent_as_png(tmp_path, mode):
    path = tmp_path / "page.tif"
    PILImage.new(mode, (40, 30)).save(path)
    image = Image(name="page", data=path)
    assert image.mime() == "image/png"
    assert decoded(image).format == "PNG"


def test_a_crop_of_a_tiff_is_sent_as_jpeg(tmp_path):
    # The label readers in hespi read crops of TIFF specimen sheets.
    path = tmp_path / "sheet.tif"
    PILImage.new("RGB", (100, 80), "white").save(path)
    crop = Image(name="sheet", data=path).make_crop(0.5, 0.5, 0.2, 0.25, name="label")
    assert crop.mime() in ACCEPTED
    sent = decoded(crop)
    assert sent.format == "JPEG" and sent.size == (20, 20)
