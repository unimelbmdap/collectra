"""Tests for collectra.utils module"""

import zipfile
from pathlib import Path

import pytest
import yaml

from collectra import Text
from collectra.utils import (
    error_msg,
    from_dir,
    get_all_files,
    get_class_path,
    is_image,
    processing_msg,
    resolve_files,
    success_msg,
    traceback_error,
    unzip,
    valid_raw_files,
)

# =============================================================================
# valid_raw_files
# =============================================================================


def test_valid_raw_files_returns_expected_extensions():
    exts = valid_raw_files()
    assert ".jpg" in exts
    assert ".png" in exts
    assert ".tiff" in exts


# =============================================================================
# resolve_files
# =============================================================================


def test_resolve_files_from_directory(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")
    result = resolve_files([tmp_path], [".jpg", ".png"])
    assert len(result) == 2


def test_resolve_files_single_file(tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"")
    result = resolve_files([f], [".jpg"])
    assert result == [f]


def test_resolve_files_no_match(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"")
    result = resolve_files([tmp_path], [".jpg"])
    assert result == []


# =============================================================================
# Message formatters
# =============================================================================


def test_success_msg():
    result = success_msg("all done")
    assert "Success" in result
    assert "all done" in result


def test_error_msg():
    result = error_msg("something broke")
    assert "Error" in result
    assert "something broke" in result


def test_processing_msg():
    result = processing_msg("working")
    assert "Processing" in result
    assert "working" in result


# =============================================================================
# get_class_path
# =============================================================================


def test_get_class_path_from_instance():
    t = Text("t", data="hello")
    assert get_class_path(t).endswith(".Text")


# =============================================================================
# get_all_files
# =============================================================================


def test_get_all_files_finds_matching_files(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.jpg").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")
    result = get_all_files([str(tmp_path)], "jpg")
    assert len(result) == 2


def test_get_all_files_raises_when_none_found(tmp_path):
    with pytest.raises(Exception):
        get_all_files([str(tmp_path)], "jpg")


def test_get_all_files_single_file(tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"")
    result = get_all_files([str(f)], "jpg")
    assert len(result) == 1


# =============================================================================
# unzip
# =============================================================================


def test_unzip_returns_dict(tmp_path):
    zip_path = tmp_path / "pipeline.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("pipeline.yaml", yaml.dump({"key": "value"}))
    result = unzip(zip_path)
    assert result == {"key": "value"}


def test_unzip_raises_on_empty_config(tmp_path):
    zip_path = tmp_path / "pipeline.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("pipeline.yaml", "")
    with pytest.raises(ValueError):
        unzip(zip_path)


# =============================================================================
# from_dir
# =============================================================================


def test_from_dir_returns_dict(tmp_path):
    (tmp_path / "pipeline.yaml").write_text(yaml.dump({"key": "value"}))
    result = from_dir(tmp_path)
    assert result == {"key": "value"}


def test_from_dir_raises_on_empty_config(tmp_path):
    (tmp_path / "pipeline.yaml").write_text("")
    with pytest.raises(ValueError):
        from_dir(tmp_path)


# =============================================================================
# traceback_error
# =============================================================================


def test_traceback_error_does_not_raise():
    traceback_error(ValueError("test error"))


def test_traceback_error_with_message_does_not_raise():
    traceback_error(ValueError("test error"), message="some context")


# =============================================================================
# is_image
# =============================================================================


def test_is_image_returns_true_for_image_extensions(tmp_path):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"]:
        assert is_image(tmp_path / f"file{ext}") is True


def test_is_image_returns_false_for_non_image():
    assert is_image(Path("file.txt")) is False
    assert is_image(Path("file.yaml")) is False
