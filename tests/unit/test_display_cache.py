import os

import pytest
from PIL import Image as PILImage

from collectra import Image
from collectra.display import DisplayContext


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "page.png"
    PILImage.new("RGB", (20, 10), (10, 20, 30)).save(path)
    return path


@pytest.fixture
def encodes(monkeypatch):
    """Counts how many times an image is actually PNG-encoded."""
    calls = []
    original = DisplayContext.publish_image

    def counting(image):
        calls.append(image.size)
        return original(image)

    monkeypatch.setattr(DisplayContext, "publish_image", staticmethod(counting))
    return calls


def view(path, rgb_view=0, **fields):
    item = Image(name="page", id="page", data=path, **fields)
    context = DisplayContext({}, path.parent, rgb_view)
    return context.image(item)


def test_showing_the_same_image_again_does_not_re_encode_it(picture, encodes):
    first = view(picture)
    second = view(picture)

    assert second["source"] == first["source"]
    assert len(encodes) == 1


def test_a_changed_file_is_encoded_again(picture, encodes):
    first = view(picture)
    PILImage.new("RGB", (20, 10), (200, 100, 50)).save(picture)
    os.utime(picture, ns=(10**18, 10**18))

    second = view(picture)

    assert second["source"] != first["source"]
    assert len(encodes) == 2


def test_a_different_orientation_is_encoded_separately(picture, encodes):
    north = view(picture, orientation="north")
    west = view(picture, orientation="west")

    assert west["source"] != north["source"]
    assert len(encodes) == 2
