"""Tests for collectra.types.images module"""

import base64
from pathlib import Path

import pytest
from PIL import Image as ImagePil

from collectra.types.images import Image, ImageCrop, Orientation


@pytest.fixture
def img_path(tmp_path):
    img = ImagePil.new("RGB", (100, 200), color=(255, 0, 0))
    path = tmp_path / "test.png"
    img.save(path)
    return path


@pytest.fixture
def image(img_path):
    return Image(name="test", data=img_path)


# =============================================================================
# Orientation
# =============================================================================


def test_orientation_to_string():
    assert Orientation.NORTH.to_string() == "north"
    assert Orientation.WEST.to_string() == "west"
    assert Orientation.SOUTH.to_string() == "south"
    assert Orientation.EAST.to_string() == "east"


def test_orientation_to_degree():
    assert Orientation.NORTH.to_degree() == 0
    assert Orientation.WEST.to_degree() == -90
    assert Orientation.SOUTH.to_degree() == -180
    assert Orientation.EAST.to_degree() == -270


def test_orientation_from_string_valid():
    assert Orientation.from_string("north") == Orientation.NORTH
    assert Orientation.from_string("west") == Orientation.WEST
    assert Orientation.from_string("south") == Orientation.SOUTH
    assert Orientation.from_string("east") == Orientation.EAST


def test_orientation_from_string_invalid():
    with pytest.raises(ValueError):
        Orientation.from_string("diagonal")


# =============================================================================
# Image construction
# =============================================================================


def test_image_loads_dimensions_and_format(img_path):
    img = Image(name="test", data=img_path)
    assert img.raw_width == 100
    assert img.raw_height == 200
    assert img.ext == "PNG"


def test_image_empty_path_raises():
    with pytest.raises(ValueError):
        Image(name="test", data="")


def test_image_nonexistent_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Image(name="test", data=tmp_path / "missing.png")


def test_image_invalid_data_type_raises():
    with pytest.raises(TypeError):
        Image(name="test", data=12345)


def test_image_string_orientation_converted(img_path):
    img = Image(name="test", data=img_path, orientation="west")
    assert img.orientation == Orientation.WEST


# =============================================================================
# Image methods
# =============================================================================


def test_image_call_returns_string_path(image, img_path):
    assert image() == str(img_path)


def test_image_get_path(image, img_path):
    assert image.get_path() == img_path


def test_image_width_height_properties(image):
    assert image.width == 100
    assert image.height == 200


def test_image_is_image_file_valid():
    assert Image.is_image_file(Path("photo.jpg")) is True
    assert Image.is_image_file(Path("photo.PNG")) is True
    assert Image.is_image_file(Path("photo.webp")) is True


def test_image_is_image_file_invalid():
    assert Image.is_image_file(Path("data.yaml")) is False
    assert Image.is_image_file(Path("notes.txt")) is False


def test_image_mime(image):
    assert image.mime() == "image/png"


def test_image_mime_no_ext(image):
    image.ext = None
    assert image.mime() == "image"


def test_image_get_encoding_is_base64(image):
    encoding = image.get_encoding()
    decoded = base64.b64decode(encoding)
    assert len(decoded) > 0


def test_image_load_returns_bytes(image):
    data = image.load()
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_image_pil_returns_pil_image(image):
    pil_img = image.pil()
    assert isinstance(pil_img, ImagePil.Image)
    assert pil_img.size == (100, 200)


def test_image_pil_rotates_for_orientation(img_path):
    img = Image(name="test", data=img_path, orientation="west")
    pil_img = img.pil()
    assert pil_img.size == (200, 100)  # width/height swapped after -90 rotation


def test_image_save_copies_file(image, tmp_path):
    dest = tmp_path / "subdir" / "copy.png"
    image.save(dest)
    assert dest.exists()


def test_image_evaluate_raises(image):
    with pytest.raises(NotImplementedError):
        image.evaluate(image)


# =============================================================================
# Image crop validation
# =============================================================================


def test_check_valid_relative_crop_values_valid(image):
    image.check_valid_relative_crop_values(0.5, 0.5, 0.5, 0.5)  # should not raise


def test_check_valid_relative_crop_values_invalid(image):
    with pytest.raises(ValueError):
        image.check_valid_relative_crop_values(1.5, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        image.check_valid_relative_crop_values(0.5, -0.1, 0.5, 0.5)
    with pytest.raises(ValueError):
        image.check_valid_relative_crop_values(0.5, 0.5, 1.5, 0.5)
    with pytest.raises(ValueError):
        image.check_valid_relative_crop_values(0.5, 0.5, 0.5, -0.1)


def test_make_crop_returns_imagecrop(image):
    crop = image.make_crop(0.5, 0.5, 0.5, 0.5)
    assert isinstance(crop, ImageCrop)
    assert crop.x_center == 0.5
    assert crop.y_center == 0.5
    assert crop.width_relative == 0.5
    assert crop.height_relative == 0.5


def test_make_crop_bounding_box(image):
    crop = image.make_crop_bounding_box(left=0, top=0, right=50, bottom=100)
    assert isinstance(crop, ImageCrop)
    assert crop.x_center == pytest.approx(0.25)
    assert crop.y_center == pytest.approx(0.25)
    assert crop.width_relative == pytest.approx(0.5)
    assert crop.height_relative == pytest.approx(0.5)


# =============================================================================
# ImageCrop
# =============================================================================


def test_imagecrop_width_height(img_path):
    crop = ImageCrop(
        name="crop",
        data=img_path,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.4,
        height_relative=0.6,
    )
    assert crop.width == 0.4 * 100
    assert crop.height == 0.6 * 200


def test_imagecrop_coordinates(img_path):
    crop = ImageCrop(
        name="crop",
        data=img_path,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.5,
        height_relative=0.5,
    )
    left, upper, right, bottom = crop.coordinates()
    assert left == 25
    assert upper == 50
    assert right == 75
    assert bottom == 150


def test_imagecrop_pil_returns_cropped_image(img_path):
    crop = ImageCrop(
        name="crop",
        data=img_path,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.5,
        height_relative=0.5,
    )
    pil_img = crop.pil()
    assert isinstance(pil_img, ImagePil.Image)
    assert pil_img.size == (50, 100)


def test_imagecrop_add_source_parent(img_path):
    crop = ImageCrop(name="crop", data=img_path)
    parent = Image(name="parent", data=img_path)
    crop.add_source_parent(parent)
    assert crop.source_parent is parent


def test_imagecrop_set_rel_to_src_parent_no_parent(img_path):
    crop = ImageCrop(name="crop", data=img_path)
    crop.set_rel_to_src_parent()  # should not raise, no-op


def test_imagecrop_set_rel_to_src_parent_with_imagecrop_parent(img_path):
    parent = ImageCrop(
        name="parent",
        data=img_path,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.5,
        height_relative=0.5,
    )
    child = ImageCrop(
        name="child",
        data=img_path,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.5,
        height_relative=0.5,
    )
    child.add_source_parent(parent)
    child.set_rel_to_src_parent()
    assert child.x_center == pytest.approx(0.5)
    assert child.y_center == pytest.approx(0.5)


def test_imagecrop_save_writes_file(img_path, tmp_path):
    crop = ImageCrop(
        name="crop",
        data=img_path,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.5,
        height_relative=0.5,
    )
    dest = tmp_path / "crop_out.png"
    crop.save(dest)
    assert dest.exists()


def test_imagecrop_save_no_path_raises(img_path):
    crop = ImageCrop(name="crop", data=img_path)
    with pytest.raises(ValueError):
        crop.save(None)


def test_imagecrop_compute_iou_identical():
    box = [0, 0, 10, 10]
    assert ImageCrop.compute_iou(box, box) == 1.0


def test_imagecrop_compute_iou_non_overlapping():
    boxA = [0, 0, 10, 10]
    boxB = [20, 20, 30, 30]
    assert ImageCrop.compute_iou(boxA, boxB) == 0.0


def test_imagecrop_compute_iou_partial():
    boxA = [0, 0, 10, 10]
    boxB = [5, 5, 15, 15]
    iou = ImageCrop.compute_iou(boxA, boxB)
    assert 0.0 < iou < 1.0


def test_imagecrop_compute_iou_zero_area():
    boxA = [0, 0, 10, 10]
    boxB = [5, 5, 5, 5]  # zero area
    assert ImageCrop.compute_iou(boxA, boxB) == 0.0


def test_imagecrop_evaluate(img_path):
    cropA = ImageCrop(
        name="a",
        data=img_path,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.5,
        height_relative=0.5,
    )
    cropB = ImageCrop(
        name="b",
        data=img_path,
        x_center=0.5,
        y_center=0.5,
        width_relative=0.5,
        height_relative=0.5,
    )
    assert cropA.evaluate(cropB) == 1.0


def test_imagecrop_evaluate_invalid_type(img_path):
    crop = ImageCrop(name="a", data=img_path)
    with pytest.raises(ValueError):
        crop.evaluate("not an imagecrop")


def test_imagecrop_make_crop_west_orientation(img_path):
    parent = ImageCrop(
        name="parent",
        data=img_path,
        x_center=0.5,
        y_center=0.5,
        width_relative=1.0,
        height_relative=1.0,
    )
    crop = parent.make_crop(0.5, 0.5, 0.5, 0.5, orientation=Orientation.WEST)
    assert isinstance(crop, ImageCrop)


def test_imagecrop_make_crop_south_orientation(img_path):
    parent = ImageCrop(
        name="parent",
        data=img_path,
        x_center=0.5,
        y_center=0.5,
        width_relative=1.0,
        height_relative=1.0,
    )
    crop = parent.make_crop(0.5, 0.5, 0.5, 0.5, orientation=Orientation.SOUTH)
    assert isinstance(crop, ImageCrop)


def test_imagecrop_make_crop_east_orientation(img_path):
    parent = ImageCrop(
        name="parent",
        data=img_path,
        x_center=0.5,
        y_center=0.5,
        width_relative=1.0,
        height_relative=1.0,
    )
    crop = parent.make_crop(0.5, 0.5, 0.5, 0.5, orientation=Orientation.EAST)
    assert isinstance(crop, ImageCrop)
