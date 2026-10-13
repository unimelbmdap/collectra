import base64
import io

import numpy as np
import pytest
from PIL import Image as PILImage

from collectra import ArtefactNode, Image, ImageSegmentation
from collectra.display import DisplayContext


@pytest.fixture
def files(tmp_path):
    image = PILImage.new("RGBA", (6, 5), (20, 40, 60, 128))
    image.save(tmp_path / "source.png")
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1:4, 2:5] = 1
    mask[2, 2] = 0
    (tmp_path / "masks").mkdir()
    PILImage.fromarray(mask).save(tmp_path / "masks/region.png")
    return tmp_path


def make(files, **kwargs):
    return ImageSegmentation(
        name="region",
        data=files / "source.png",
        mask=files / "masks/region.png",
        **kwargs,
    )


def test_cropping_alpha_and_source_unchanged(files):
    item = make(files)
    assert isinstance(item, Image)
    assert item.display_bounds() == (2, 1, 5, 4)
    assert (item.width, item.height) == (3, 3)
    result = item.pil()
    assert result.mode == "RGBA"
    assert result.size == (3, 3)
    assert result.getpixel((0, 0)) == (20, 40, 60, 128)
    assert result.getpixel((0, 1))[3] == 0
    assert PILImage.open(files / "source.png").size == (6, 5)
    assert np.asarray(PILImage.open(files / "masks/region.png")).max() == 1


@pytest.mark.parametrize("orientation", ["north", "west", "south", "east"])
def test_orientation(files, orientation):
    original = make(files).pil()
    item = make(files, orientation=orientation)
    expected = original.rotate(item.orientation.to_degree(), expand=True)
    assert item.pil().tobytes() == expected.tobytes()


def test_serialization_loader_and_display(files, monkeypatch):
    monkeypatch.chdir(files)
    item = ImageSegmentation(
        name="region", id="region", data="source.png", mask="masks/region.png"
    )
    record = item.serialize()
    assert record["type"] == "collectra.ImageSegmentation"
    assert record["mask"] == "masks/region.png"
    import yaml

    (files / "results.yaml").write_text(yaml.safe_dump({"region": record}))
    node = ArtefactNode(name="region", types={ImageSegmentation})
    loaded = ArtefactNode.batch_process(files, [node])
    assert len(loaded) == 1
    assert loaded[0].pil().tobytes() == item.pil().tobytes()
    monkeypatch.chdir(files.parent)
    context = DisplayContext({"region": record}, files)
    view = context.artefact("region").display(context)
    preview = PILImage.open(io.BytesIO(base64.b64decode(view["source"].split(",")[1])))
    assert preview.tobytes() == item.pil().tobytes()


def test_save_extract_and_encoding(files):
    item = make(files)
    item.save(files / "saved.png")
    item.extract(files / "extracted")
    for path in ["saved.png", "extracted.png"]:
        assert PILImage.open(files / path).tobytes() == item.pil().tobytes()
    assert item.mime() == "image/png"
    encoded = PILImage.open(io.BytesIO(base64.b64decode(item.get_encoding())))
    assert encoded.tobytes() == item.pil().tobytes()


@pytest.mark.parametrize("kind", ["empty", "size", "values", "rgb", "format"])
def test_invalid_masks(files, kind):
    mask = PILImage.new("L", (6, 5), 1)
    if kind == "empty":
        mask = PILImage.new("L", (6, 5), 0)
    elif kind == "size":
        mask = PILImage.new("L", (3, 2), 1)
    elif kind == "values":
        mask.putpixel((0, 0), 2)
    elif kind == "rgb":
        mask = mask.convert("RGB")
    mask.save(files / "masks/region.png", format="BMP" if kind == "format" else "PNG")
    with pytest.raises(ValueError):
        make(files)


def test_one_bit_mask(files):
    mask = PILImage.new("1", (6, 5), 0)
    mask.putpixel((2, 1), 1)
    mask.save(files / "masks/region.png")
    assert make(files).pil().size == (1, 1)


def test_cli_extract_uses_png(files):
    from collectra.types.images import ImageArtefactCommands

    item = make(files, id="segment")
    node = ArtefactNode(name="region", items={item.id: item}, types={ImageSegmentation})
    ImageArtefactCommands(node, "yaml").extract(output=files / "output")
    extracted = PILImage.open(files / "output/segment.png")
    assert extracted.tobytes() == item.pil().tobytes()
