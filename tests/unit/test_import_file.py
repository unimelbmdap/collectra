"""Importing one input file as a results folder next to it."""

import shutil
from pathlib import Path

import pytest
import yaml

from collectra import Pipeline
from collectra.commons.results import pop_header
from collectra.pipelines.importing import destination
from collectra.tasks.base import Task
from collectra.types.images import Image
from collectra.types.texts import Text

THIS = "tests.unit.test_import_file"


class Describe(Task):
    def run(self, image: Image) -> Text:
        return Text(name=self.get_output_name(), data="page")


def build(tmp_path):
    return Pipeline(
        "Import",
        "imp",
        "0.1.0",
        path=str(tmp_path),
        describe={"type": f"{THIS}.Describe", "input": ["page"], "output": ["text"]},
    )


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    return folder


def snapshot(folder: Path) -> dict:
    return {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in folder.rglob("*")
        if path.is_file()
    }


def test_import_makes_folder_copy_and_root_only_results(tmp_path, source, raw_img_path):
    image = source / "page001.jpg"
    shutil.copy(raw_img_path, image)
    folder = build(tmp_path).import_file(image)
    assert folder == source / "page001.imp"
    assert sorted(path.name for path in folder.iterdir()) == [
        "page001.jpg",
        "results.yaml",
    ]
    assert (folder / "page001.jpg").read_bytes() == image.read_bytes()
    results = yaml.safe_load((folder / "results.yaml").read_text())
    header = pop_header(results)
    assert (header.workflow, header.version) == ("Import", "0.1.0")
    assert results == {
        "page": {"type": "collectra.Image", "id": "page", "data": "page001.jpg"}
    }


def test_same_source_is_not_imported_twice(tmp_path, source, raw_img_path):
    image = source / "page.jpg"
    shutil.copy(raw_img_path, image)
    pipeline = build(tmp_path)
    folder = pipeline.import_file(image)
    before = snapshot(source)
    assert pipeline.import_file(image) == folder
    assert not (source / "page-2.imp").exists()
    assert snapshot(source) == before


def test_different_file_with_same_stem_is_numbered(tmp_path, source, raw_img_path):
    shutil.copy(raw_img_path, source / "page.jpg")
    shutil.copy(raw_img_path, source / "page.png")
    pipeline = build(tmp_path)
    assert pipeline.import_file(source / "page.jpg") == source / "page.imp"
    assert pipeline.import_file(source / "page.png") == source / "page-2.imp"
    before = snapshot(source / "page.imp")
    (source / "page.jpg").write_bytes(raw_img_path.read_bytes() + b"other")
    assert pipeline.import_file(source / "page.jpg") == source / "page-3.imp"
    assert snapshot(source / "page.imp") == before


def test_page_missing_its_copy_is_not_reused(tmp_path, source, raw_img_path):
    shutil.copy(raw_img_path, source / "page.jpg")
    pipeline = build(tmp_path)
    (pipeline.import_file(source / "page.jpg") / "page.jpg").unlink()
    assert pipeline.import_file(source / "page.jpg") == source / "page-2.imp"


@pytest.mark.parametrize(
    "text",
    [
        "page: [",
        "- page\n- other\n",
        "42\n",
        "results: {type: collectra.Results, workflow: [1]}\n",
        "page: [1, 2]\n",
    ],
    ids=["invalid", "list", "scalar", "bad-header", "bad-entry"],
)
def test_unreadable_results_are_not_this_page(tmp_path, source, raw_img_path, text):
    shutil.copy(raw_img_path, source / "page.jpg")
    (source / "page.imp").mkdir()
    shutil.copy(raw_img_path, source / "page.imp" / "page.jpg")
    (source / "page.imp" / "results.yaml").write_text(text)
    before = snapshot(source / "page.imp")
    assert build(tmp_path).import_file(source / "page.jpg") == source / "page-2.imp"
    assert snapshot(source / "page.imp") == before


def test_name_comes_from_the_stem(tmp_path, source, raw_img_path):
    shutil.copy(raw_img_path, source / "a.jpg.jpg")
    shutil.copy(raw_img_path, source / "PAGE.JPG")
    pipeline = build(tmp_path)
    assert pipeline.import_file(source / "a.jpg.jpg") == source / "a.jpg.imp"
    assert pipeline.import_file(source / "PAGE.JPG") == source / "PAGE.imp"
    assert (source / "PAGE.imp" / "PAGE.JPG").exists()


@pytest.mark.parametrize("name", ["notes.txt", "page.imp"])
def test_unaccepted_extension_is_refused(tmp_path, source, name):
    (source / name).write_text("not an image")
    with pytest.raises(ValueError, match=name):
        build(tmp_path).import_file(source / name)
    assert sorted(source.iterdir()) == [source / name]


def test_failed_copy_leaves_nothing(tmp_path, source, raw_img_path, monkeypatch):
    shutil.copy(raw_img_path, source / "page.jpg")
    # A folder that was already there is taken, and is kept on failure.
    (source / "page.imp").mkdir()
    (source / "page.imp" / "mine.txt").write_text("kept")

    def fail(*args, **kwargs):
        assert (source / "page-2.imp").is_dir()
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", fail)
    with pytest.raises(OSError, match="disk full"):
        build(tmp_path).import_file(source / "page.jpg")
    assert sorted(path.name for path in source.iterdir()) == ["page.imp", "page.jpg"]
    assert (source / "page.imp" / "mine.txt").read_text() == "kept"


def test_destination_is_the_first_name_not_taken():
    assert destination(Path("/in/page.jpg"), "imp", lambda path: False) == Path(
        "/in/page.imp"
    )


def test_destination_numbers_taken_names():
    taken = {Path("/in/page.imp"), Path("/in/page-2.imp")}
    assert destination(Path("/in/page.jpg"), "imp", taken.__contains__) == Path(
        "/in/page-3.imp"
    )


def test_destination_consults_taken_per_candidate():
    asked = []

    def taken(path):
        asked.append(path)
        return len(asked) < 3

    assert destination(Path("/in/a.jpg.jpg"), "imp", taken) == Path("/in/a.jpg-3.imp")
    assert asked == [
        Path("/in/a.jpg.imp"),
        Path("/in/a.jpg-2.imp"),
        Path("/in/a.jpg-3.imp"),
    ]
