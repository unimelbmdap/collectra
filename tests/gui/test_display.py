import base64
import io
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image as PILImage

from collectra import Image, ImageCrop, Text
from collectra.gui.backend import GUIBackend
from collectra.types.base import Artefact


@dataclass
class Measurement(Artefact):
    value: float = 0

    def __call__(self):
        return self.value

    def display(self, context):
        return {
            "kind": "text",
            "format": "plain",
            "text": str(self.value),
            "editable": False,
        }


class SpectralTile(Image):
    pass


class Region(ImageCrop):
    pass


def record(kind, id, **kwargs):
    return dict(type=kind, id=id, **kwargs)


@pytest.fixture
def backend(tmp_path):
    PILImage.new("RGB", (200, 100), "red").save(tmp_path / "image.png")
    PILImage.new("RGB", (100, 100), "blue").save(tmp_path / "other.png")
    data = {
        "image": record(
            "tests.gui.test_display.SpectralTile", "image", data="image.png"
        ),
        "crop": record(
            "tests.gui.test_display.Region",
            "crop",
            data="image.png",
            parents="image",
            x_center=0.5,
            y_center=0.5,
            width_relative=0.5,
            height_relative=0.6,
        ),
        "child": record(
            "collectra.ImageCrop",
            "child",
            data="image.png",
            parents="crop",
            x_center=0.4,
            y_center=0.4,
            width_relative=0.1,
            height_relative=0.2,
        ),
        "different": record(
            "collectra.ImageCrop",
            "different",
            data="other.png",
            parents="image",
            x_center=0.5,
            y_center=0.5,
            width_relative=0.2,
            height_relative=0.2,
        ),
        "link": record("collectra.Link", "link", parents="crop"),
        "measurement": record(
            "tests.gui.test_display.Measurement", "measurement", value=12.5, units="mm"
        ),
        "text": record("collectra.Text", "text", data="inline text"),
    }
    (tmp_path / "results.yaml").write_text(yaml.safe_dump(data))
    backend = GUIBackend(SimpleNamespace(ext="collectra"))
    assert backend.load_yaml(str(tmp_path))["success"]
    return backend


def view(backend, id):
    result = backend.get_artefact_display(id)
    assert result["success"], result
    return result["view"]


def pixels(view):
    return PILImage.open(io.BytesIO(base64.b64decode(view["source"].split(",", 1)[1])))


def test_inherited_image_display_and_immediate_same_source_children(backend):
    result = view(backend, "image")
    assert pixels(result).size == (200, 100)
    assert [r["id"] for r in result["annotations"]] == ["crop"]
    assert result["can_create_crop"]
    assert Region.display is Image.display


def test_crop_and_link_display(backend):
    result = view(backend, "crop")
    assert pixels(result).size == (100, 60)
    assert [r["id"] for r in result["annotations"]] == ["child"]
    assert result["annotations"][0]["crop_region"] == pytest.approx(
        dict(x_center=0.3, y_center=1 / 3, width_relative=0.2, height_relative=1 / 3)
    )
    assert view(backend, "link") == result


@pytest.mark.parametrize("orientation", ["north", "west", "south", "east"])
def test_create_and_edit_in_rotated_crop_coordinates(backend, orientation):
    backend._graph.get_node("crop").orientation = orientation
    region = dict(x_center=0.3, y_center=0.4, width_relative=0.2, height_relative=0.3)
    result = backend.create_annotation(
        region, "new", "crop", "image.png", crop_id="new", view_id="crop"
    )
    assert result["success"], result
    rows = backend.get_display_annotations("crop")["rows"]
    assert next(r for r in rows if r["id"] == "new")["crop_region"] == pytest.approx(
        region
    )
    region["x_center"] = 0.6
    assert backend.update_node_coordinates("new", region, "crop")["success"]
    rows = backend.get_display_annotations("crop")["rows"]
    assert next(r for r in rows if r["id"] == "new")["crop_region"] == pytest.approx(
        region
    )
    assert backend._graph.get_node("new").parents == ["crop"]
    preview = pixels(view(backend, "crop"))
    assert preview.size == ((60, 100) if orientation in ("east", "west") else (100, 60))


def test_custom_type_and_roundtrip(backend):
    assert view(backend, "measurement")["text"] == "12.5"
    assert not backend.update_display_text("measurement", "invalid")["success"]
    assert backend.update_display_text("text", "edited")["success"]
    saved = yaml.safe_load(Path(backend._yaml_path).read_text())
    assert saved["measurement"]["units"] == "mm"
    assert saved["measurement"]["value"] == 12.5
    assert saved["text"]["data"] == "edited"


@pytest.mark.parametrize(
    ("suffix", "format"), [(".md", "markdown"), (".xml", "xml"), (".txt", "plain")]
)
def test_file_text_display_and_save_preserves_reference(backend, suffix, format):
    path = Path(backend._yaml_path).parent / ("custom-name" + suffix)
    path.write_text("old text")
    backend._graph.set_data("text", path.name)
    assert view(backend, "text")["format"] == format
    assert view(backend, "text")["text"] == "old text"
    assert backend.update_display_text("text", "new text")["success"]
    assert path.read_text() == "new text"
    assert backend._graph.get_data("text") == path.name


def test_inline_text_named_markdown_is_plain(backend):
    backend._graph.rename_id("text", "markdown")
    assert view(backend, "markdown")["format"] == "plain"


def test_link_cycle_reports_error(backend):
    backend._graph.add_node(
        dict(label="loop", type="collectra.Link", id="loop", parents="loop")
    )
    result = backend.get_artefact_display("loop")
    assert not result["success"]
    assert "Cyclic Link" in result["error"]


def test_rotated_child_box_has_expected_position(backend):
    backend._graph.get_node("crop").orientation = "west"
    region = view(backend, "crop")["annotations"][0]["crop_region"]
    assert region == pytest.approx(
        dict(x_center=2 / 3, y_center=0.3, width_relative=1 / 3, height_relative=0.2)
    )


def test_linked_text_edits_target(backend):
    backend._graph.add_node(
        dict(label="alias", type="collectra.Link", id="alias", parents="text")
    )
    assert view(backend, "alias")["target_id"] == "text"
    assert backend.update_display_text("alias", "through link")["success"]
    assert backend._graph.get_data("text") == "through link"
    assert backend._graph.get_type("alias") == "collectra.Link"


def test_tiff_preview_preserves_source(backend):
    import numpy as np
    import tifffile

    path = Path(backend._yaml_path).parent / "views.tif"
    bands = np.zeros((75, 10, 20), dtype=np.uint8)
    bands[0] = 255
    tifffile.imwrite(path, bands, photometric="minisblack", metadata={"axes": "CYX"})
    original = path.read_bytes()
    backend._graph.add_node(
        dict(label="views", type="collectra.Image", id="views", data=path.name)
    )
    preview = pixels(view(backend, "views"))
    assert preview.size == (20, 10)
    assert preview.getpixel((0, 0)) == (255, 0, 0)
    assert path.read_bytes() == original


def test_folder_with_text_only_can_be_opened(backend, tmp_path):
    folder = tmp_path / "text.collectra"
    folder.mkdir()
    (folder / "results.yaml").write_text(
        "text: {type: collectra.Text, id: text, data: Hello}"
    )
    assert backend._scan_collectra_folder(folder)["image_path"] == ""
    assert backend.load_yaml(str(folder))["success"]
    assert view(backend, "text")["text"] == "Hello"


@pytest.mark.parametrize("axes", ["CYX", "YXC"])
@pytest.mark.parametrize("target", ["views", "view-crop", "view-link"])
def test_rgb_views_preserve_crop_geometry_and_select_triplets(backend, axes, target):
    import numpy as np
    import tifffile

    path = Path(backend._yaml_path).parent / "views.tif"
    bands = np.broadcast_to(np.arange(75, dtype=np.uint8), (10, 20, 75)).copy()
    tifffile.imwrite(
        path,
        bands.transpose(2, 0, 1) if axes == "CYX" else bands,
        photometric="minisblack",
        metadata={"axes": axes},
    )
    original = path.read_bytes()
    backend._graph.add_node(
        dict(label="views", type="collectra.Image", id="views", data=path.name)
    )
    backend._graph.add_node(
        dict(
            label="view-crop",
            type="collectra.ImageCrop",
            id="view-crop",
            parents="views",
            data=path.name,
            x_center=0.5,
            y_center=0.5,
            width_relative=0.5,
            height_relative=0.6,
            orientation="west",
        )
    )
    backend._graph.add_node(
        dict(
            label="view-link",
            type="collectra.Link",
            id="view-link",
            parents="view-crop",
        )
    )
    first = view(backend, target)
    assert first["rgb_view_count"] == 25
    assert first["rgb_view_index"] == 0
    assert pixels(first).getpixel((0, 0)) == (0, 1, 2)
    for index in (1, 24, 0):
        result = backend.get_artefact_display(target, index)
        assert result["success"], result
        selected = result["view"]
        assert selected["rgb_view_index"] == index
        assert selected["annotations"] == first["annotations"]
        assert pixels(selected).size == ((20, 10) if target == "views" else (6, 10))
        assert pixels(selected).getpixel((0, 0)) == tuple(
            range(index * 3, index * 3 + 3)
        )
    assert path.read_bytes() == original


def test_rgb_page_stack_decodes_only_selected_three_pages(backend, monkeypatch):
    import numpy as np
    import tifffile

    path = Path(backend._yaml_path).parent / "stack.tif"
    tifffile.imwrite(
        path,
        np.zeros((75, 10, 20), dtype=np.uint8),
        photometric="minisblack",
        metadata={"axes": "CYX"},
    )
    backend._graph.add_node(
        dict(label="stack", type="collectra.Image", id="stack", data=path.name)
    )
    decoded = []
    read = tifffile.TiffPage.asarray

    def capture(page, *args, **kwargs):
        decoded.append(page.index)
        return read(page, *args, **kwargs)

    monkeypatch.setattr(tifffile.TiffPage, "asarray", capture)
    result = backend.get_artefact_display("stack", 24)
    assert result["success"], result
    assert decoded == [72, 73, 74]


@pytest.mark.parametrize("index", [-1, 25, 0.5, "1", True])
def test_invalid_rgb_view_returns_error(backend, index):
    result = backend.get_artefact_display("image", index)
    assert not result["success"]
    assert "RGB view" in result["error"]


def test_ordinary_image_has_one_view(backend):
    result = view(backend, "image")
    assert result["rgb_view_count"] == 1
    assert result["rgb_view_index"] == 0


def test_gui_startup_inputs_preserve_order_and_deduplicate(tmp_path):
    folders = []
    for name in ("b.collectra", "a.collectra"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "results.yaml").write_text(
            "text: {type: collectra.Text, id: text, data: hello}"
        )
        folders.append(folder)
    api = GUIBackend(
        SimpleNamespace(ext="collectra"),
        inputs=[
            folders[0] / "results.yaml",
            tmp_path,
            folders[0],
            folders[1] / "results.yaml",
        ],
    )
    initial = api.get_initial_items()
    assert initial["success"] and initial["provided"]
    assert [f["name"] for f in initial["folders"]] == ["b.collectra", "a.collectra"]
    assert [f["index"] for f in initial["folders"]] == [0, 1]
    assert api.load_collectra_folder(0)["folder_path"] == str(folders[0])
    assert api.load_yaml(api.load_collectra_folder(1)["yaml_path"])["success"]
    assert api.get_artefact_display("text")["view"]["text"] == "hello"


def test_gui_startup_without_inputs_keeps_picker(backend):
    assert backend.get_initial_items() == {"success": True, "provided": False}


def test_gui_startup_reports_invalid_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="GUI input does not exist"):
        GUIBackend(SimpleNamespace(ext="collectra"), inputs=[tmp_path / "missing"])
    with pytest.raises(ValueError, match="No GUI results found"):
        GUIBackend(SimpleNamespace(ext="collectra"), inputs=[tmp_path])
