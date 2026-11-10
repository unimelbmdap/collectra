import pytest

from collectra import Image, ImageCrop, Link, Text
from collectra.types.base import Artefact


def test_the_base_artefact_does_not_know_its_extensions():
    with pytest.raises(NotImplementedError, match="file extensions"):
        Artefact.file_extensions()


def test_an_image_lists_its_extensions():
    extensions = Image.file_extensions()
    assert isinstance(extensions, set)
    assert {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"} <= extensions
    assert ImageCrop.file_extensions() == extensions


def test_a_text_lists_its_extensions():
    extensions = Text.file_extensions()
    assert isinstance(extensions, set)
    assert {".txt", ".md", ".xml"} <= extensions
    assert all(extension.startswith(".") for extension in extensions)


def test_a_link_owns_no_files():
    assert Link.file_extensions() == set()
