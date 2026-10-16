import base64
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from collectra.gui.backend import GUIBackend


@pytest.fixture
def page(tmp_path):
    Image.new("RGB", (40, 30), (100, 150, 200)).save(tmp_path / "page.png")
    mask = np.zeros((30, 40), dtype="uint8")
    mask[10:20, 10:20] = 1
    Image.fromarray(mask).save(tmp_path / "mask.png")
    results = tmp_path / "results.yaml"
    results.write_text(
        yaml.safe_dump(
            {
                "region": {
                    "type": "collectra.ImageSegmentation",
                    "id": "region",
                    "data": "page.png",
                    "mask": "mask.png",
                }
            }
        )
    )
    backend = GUIBackend(SimpleNamespace(ext="collectra"))
    assert backend.load_yaml(str(results))["success"]
    return backend, results


def decode(source):
    return Image.open(io.BytesIO(base64.b64decode(source.split(",")[1])))


@pytest.mark.parametrize("orientation", ["north", "west", "south", "east"])
def test_full_source_preview_and_editable_mask_are_aligned(page, orientation):
    backend, results = page
    data = yaml.safe_load(results.read_text())
    data["region"]["orientation"] = orientation
    results.write_text(yaml.safe_dump(data))
    backend.load_yaml(str(results))
    view = backend.get_artefact_display("region")["view"]
    source = decode(view["source"])
    mask = decode(view["segmentation"]["mask_source"])
    assert source.size == mask.size
    assert source.size == ((40, 30) if orientation in ("north", "south") else (30, 40))
    assert source.getpixel((0, 0))[:3] == (100, 150, 200)
    assert mask.getchannel("A").getpixel((0, 0)) == 0
    assert set(np.asarray(mask.getchannel("A")).ravel()) == {0, 255}
    assert view["segmentation"]["orientation"] == orientation


@pytest.mark.parametrize("orientation", ["north", "west", "south", "east"])
def test_paint_in_rotated_view_saves_source_sized_binary_mask(page, orientation):
    backend, results = page
    data = yaml.safe_load(results.read_text())
    data["region"]["orientation"] = orientation
    results.write_text(yaml.safe_dump(data))
    backend.load_yaml(str(results))
    source = (results.parent / "page.png").read_bytes()
    yaml_before = results.read_bytes()
    strokes = [{"tool": "brush", "radius": 2, "points": [[4, 4], [8, 4]]}]
    response = backend.update_segmentation_mask(
        "region", strokes, str(results), orientation
    )
    assert response["success"], response
    assert response["changed"]
    mask = Image.open(results.parent / "mask.png")
    assert mask.size == (40, 30)
    assert set(np.asarray(mask).ravel()) == {0, 1}
    from collectra.types.images import Orientation

    rotated = mask.rotate(Orientation.from_string(orientation).to_degree(), expand=True)
    assert rotated.getpixel((4, 4)) == 1
    assert rotated.getpixel((8, 4)) == 1
    assert rotated.getpixel((4, 8)) == 0  # A circle, not its bounding square.
    assert (results.parent / "page.png").read_bytes() == source
    assert results.read_bytes() == yaml_before


def test_eraser_and_unchanged_strokes(page):
    backend, results = page
    strokes = [{"tool": "eraser", "radius": 2, "points": [[15, 15]]}]
    assert backend.update_segmentation_mask("region", strokes, str(results), "north")[
        "changed"
    ]
    mask = Image.open(results.parent / "mask.png")
    assert mask.getpixel((15, 15)) == 0
    assert mask.getpixel((10, 10)) == 1
    before = (results.parent / "mask.png").stat().st_mtime_ns
    assert not backend.update_segmentation_mask(
        "region", strokes, str(results), "north"
    )["changed"]
    assert (results.parent / "mask.png").stat().st_mtime_ns == before


@pytest.mark.parametrize(
    "strokes,orientation",
    [
        ([{"tool": "eraser", "radius": 100, "points": [[15, 15]]}], "north"),
        ([{"tool": "brush", "radius": float("nan"), "points": [[1, 1]]}], "north"),
        ([{"tool": "brush", "radius": 2, "points": [[float("inf"), 1]]}], "north"),
        ([{"tool": "unknown", "radius": 2, "points": [[1, 1]]}], "north"),
        ([{"tool": "brush", "radius": 2, "points": [[1, 1]]}], "west"),
    ],
)
def test_invalid_edits_leave_mask_intact(page, strokes, orientation):
    backend, results = page
    before = (results.parent / "mask.png").read_bytes()
    response = backend.update_segmentation_mask(
        "region", strokes, str(results), orientation
    )
    assert not response["success"]
    assert (results.parent / "mask.png").read_bytes() == before


def test_late_save_targets_original_page(page, tmp_path):
    backend, results = page
    other = tmp_path / "other"
    other.mkdir()
    (other / "results.yaml").write_text(
        yaml.safe_dump(
            {"text": {"type": "collectra.Text", "id": "text", "data": "other"}}
        )
    )
    backend.load_yaml(str(other))
    response = backend.update_segmentation_mask(
        "region",
        [{"tool": "brush", "radius": 2, "points": [[4, 4]]}],
        str(results),
        "north",
    )
    assert response["success"], response
    assert Image.open(results.parent / "mask.png").getpixel((4, 4)) == 1
    assert backend.get_artefact_display("text")["view"]["text"] == "other"


def test_failed_atomic_save_preserves_mask(page, monkeypatch):
    backend, results = page
    import collectra.gui.segmentation as editing

    def fail(*args):
        raise OSError("Disk full")

    monkeypatch.setattr(editing.os, "replace", fail)
    before = (results.parent / "mask.png").read_bytes()
    response = backend.update_segmentation_mask(
        "region",
        [{"tool": "brush", "radius": 2, "points": [[4, 4]]}],
        str(results),
        "north",
    )
    assert not response["success"]
    assert "Disk full" in response["error"]
    assert (results.parent / "mask.png").read_bytes() == before
    assert not list(results.parent.glob(".mask-*.png"))
