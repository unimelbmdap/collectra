import base64
import io

import numpy as np
import pytest
from PIL import Image

from collectra.display import DisplayContext


@pytest.mark.parametrize("parent_type", ["Image", "ImageCrop", "ImageSegmentation"])
@pytest.mark.parametrize("orientation", ["north", "west"])
def test_child_mask_matches_parent_display(tmp_path, parent_type, orientation):
    Image.new("RGB", (20, 10)).save(tmp_path / "page.png")
    mask = np.zeros((10, 20), dtype="uint8")
    mask[2:8, 5:15] = 1
    mask[4:6, 8:10] = 0  # Preserve the hole, rather than drawing a bounding box.
    Image.fromarray(mask).save(tmp_path / "child.png")
    parent_mask = np.zeros_like(mask)
    parent_mask[1:9, 4:12] = 1
    Image.fromarray(parent_mask).save(tmp_path / "parent.png")
    parent = dict(
        type=f"collectra.{parent_type}",
        id="page",
        data="page.png",
        orientation=orientation,
    )
    if parent_type == "ImageCrop":
        parent.update(
            x_center=0.4, y_center=0.5, width_relative=0.4, height_relative=0.8
        )
    elif parent_type == "ImageSegmentation":
        parent["mask"] = "parent.png"
    records = {
        "page": parent,
        "child": dict(
            type="collectra.ImageSegmentation",
            id="child",
            label="lines",
            name="First line",
            data="page.png",
            mask="child.png",
            parents="page",
        ),
        "unrelated": dict(
            type="collectra.ImageSegmentation",
            id="unrelated",
            data="page.png",
            mask="child.png",
            parents="other",
        ),
    }
    context = DisplayContext(records, tmp_path)
    item = context.artefact("page")
    rows = context.annotations(item)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "child"
    assert row["label"] == "lines"
    assert row["name"] == "First line"
    overlay = Image.open(io.BytesIO(base64.b64decode(row["mask_source"].split(",")[1])))
    expected = Image.fromarray(mask * 255)
    if parent_type == "ImageSegmentation":
        expected = Image.fromarray(mask * parent_mask * 255)
    expected = expected.crop(item.display_bounds()).rotate(
        item.orientation.to_degree(), expand=True
    )
    bounds = expected.getbbox()
    assert np.array_equal(
        np.asarray(overlay.getchannel("A")), np.asarray(expected.crop(bounds))
    )
    x, y, right, bottom = bounds
    assert row["mask_region"] == dict(
        x=x / expected.width,
        y=y / expected.height,
        width=(right - x) / expected.width,
        height=(bottom - y) / expected.height,
    )
