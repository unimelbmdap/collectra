"""Importing the inputs a user opens as results folders."""

import shutil
from pathlib import Path

import pytest
import yaml

from collectra import Pipeline
from collectra.commons.results import pop_header
from collectra.pipelines.importing import ImportReport
from collectra.tasks.base import Task
from collectra.types.images import Image
from collectra.types.texts import Text

THIS = "tests.unit.test_import_inputs"


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
        path: (None if path.is_dir() else path.read_bytes(), path.stat().st_mtime_ns)
        for path in [folder, *folder.rglob("*")]
    }


def test_existing_pages_only_write_nothing(tmp_path, source, raw_img_path):
    pipeline = build(tmp_path)
    for name in ("a.jpg", "b.jpg"):
        shutil.copy(raw_img_path, source / name)
        pipeline.import_file(source / name)
    (source / ".DS_Store").write_text("x")
    before = snapshot(source)
    report = pipeline.import_inputs([source])
    assert report.pages == [source / "a.imp", source / "b.imp"]
    assert (report.imported, report.existing) == (0, 2)
    assert report.summary("Import") == ""
    assert snapshot(source) == before


def test_a_mix_is_counted_and_summarised(tmp_path, source, raw_img_path):
    pipeline = build(tmp_path)
    shutil.copy(raw_img_path, source / "b.jpg")
    shutil.copy(raw_img_path, source / "c.PNG")
    shutil.copy(raw_img_path, source / "old.jpg")
    pipeline.import_file(source / "old.jpg")
    (source / "notes.txt").write_text("text")
    (source / ".DS_Store").write_text("x")
    (source / "sub").mkdir()
    (source / "sub" / "deep.jpg").write_bytes(raw_img_path.read_bytes())
    report = pipeline.import_inputs([source])
    assert report.pages == [source / "b.imp", source / "c.imp", source / "old.imp"]
    assert (report.imported, report.existing) == (2, 1)
    assert report.skipped == ["notes.txt: not an input the Import pipeline accepts"]
    assert report.failed == []
    assert report.summary("Grapto") == (
        "Imported 2 pages, opened 1 existing, skipped 1 file Grapto doesn't accept"
    )
    assert not (source / "sub" / "deep.imp").exists()
    assert sorted(path.name for path in (source / "sub").iterdir()) == ["deep.jpg"]


def test_adopts_a_folder_with_one_image_in_place(tmp_path, source, raw_img_path):
    folder = source / "A.imp"
    folder.mkdir()
    shutil.copy(raw_img_path, folder / "scan.jpg")
    report = build(tmp_path).import_inputs([source])
    assert report.pages == [folder]
    assert (report.imported, report.existing) == (1, 0)
    assert sorted(path.name for path in folder.iterdir()) == [
        "results.yaml",
        "scan.jpg",
    ]
    results = yaml.safe_load((folder / "results.yaml").read_text())
    pop_header(results)
    assert results == {
        "page": {"type": "collectra.Image", "id": "page", "data": "scan.jpg"}
    }


@pytest.mark.parametrize("names", [[], ["a.jpg", "b.png"]], ids=["none", "several"])
def test_does_not_adopt_without_exactly_one_image(
    tmp_path, source, raw_img_path, names
):
    folder = source / "A.imp"
    folder.mkdir()
    for name in names:
        shutil.copy(raw_img_path, folder / name)
    (folder / ".hidden.jpg").write_text("x")
    before = snapshot(folder)
    report = build(tmp_path).import_inputs([source])
    assert report.pages == []
    assert report.imported == 0
    assert report.summary("Import") == "skipped 1 folder without exactly one image"
    assert snapshot(folder) == before


def test_an_existing_results_file_is_opened_as_is(tmp_path, source, raw_img_path):
    folder = source / "A.imp"
    folder.mkdir()
    shutil.copy(raw_img_path, folder / "scan.jpg")
    (folder / "results.yaml").write_text("mine: 1\n")
    before = snapshot(folder)
    report = build(tmp_path).import_inputs([folder])
    assert (report.pages, report.existing, report.imported) == ([folder], 1, 0)
    assert snapshot(folder) == before


def test_failed_adoption_removes_only_what_it_created(
    tmp_path, source, raw_img_path, monkeypatch
):
    folder = source / "A.imp"
    folder.mkdir()
    shutil.copy(raw_img_path, folder / "scan.jpg")
    pipeline = build(tmp_path)

    def fail(directory, key, value):
        (directory / "results.yaml").write_text("partial")
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "_create_results_file", fail)
    report = pipeline.import_inputs([folder])
    assert report.failed == ["A.imp: disk full"]
    assert (report.pages, report.imported) == ([], 0)
    assert sorted(path.name for path in folder.iterdir()) == ["scan.jpg"]
    assert report.summary("Import") == "1 failed"


def test_one_failing_file_does_not_stop_the_others(
    tmp_path, source, raw_img_path, monkeypatch
):
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        shutil.copy(raw_img_path, source / name)
    pipeline = build(tmp_path)
    original = pipeline.import_file

    def flaky(path):
        if path.name == "b.jpg":
            raise OSError("read-only")
        return original(path)

    monkeypatch.setattr(pipeline, "import_file", flaky)
    report = pipeline.import_inputs([source])
    assert report.pages == [source / "a.imp", source / "c.imp"]
    assert (report.imported, report.failed) == (2, ["b.jpg: read-only"])
    assert report.summary("Import") == "Imported 2 pages, 1 failed"


def test_a_file_and_a_page_folder_are_inputs(tmp_path, source, raw_img_path):
    pipeline = build(tmp_path)
    shutil.copy(raw_img_path, source / "one.jpg")
    shutil.copy(raw_img_path, source / "two.jpg")
    pipeline.import_file(source / "two.jpg")
    report = pipeline.import_inputs([source / "one.jpg", source / "two.imp"])
    assert report.pages == [source / "one.imp", source / "two.imp"]
    assert (report.imported, report.existing) == (1, 1)
    assert report.summary("Import") == "Imported 1 page, opened 1 existing"


def test_children_are_taken_in_case_insensitive_order(tmp_path, source, raw_img_path):
    for name in ("b.jpg", "A.jpg", "C.jpg"):
        shutil.copy(raw_img_path, source / name)
    report = build(tmp_path).import_inputs([source])
    assert [page.name for page in report.pages] == ["A.imp", "b.imp", "C.imp"]


def test_a_missing_input_raises(tmp_path, source):
    with pytest.raises(FileNotFoundError, match="gone"):
        build(tmp_path).import_inputs([source / "gone"])


def test_report_summary_parts():
    report = ImportReport(
        imported=1, existing=2, skipped=["a: x", "b: x"], failed=["c: y"]
    )
    assert report.summary("Grapto") == (
        "Imported 1 page, opened 2 existing, skipped 2 files Grapto doesn't accept, "
        "1 failed"
    )
    assert ImportReport().summary("Grapto") == ""


def test_an_already_imported_file_is_opened(tmp_path, source, raw_img_path):
    pipeline = build(tmp_path)
    shutil.copy(raw_img_path, source / "one.jpg")
    folder = pipeline.import_file(source / "one.jpg")
    before = snapshot(source)
    report = pipeline.import_inputs([source / "one.jpg"])
    assert (report.pages, report.imported, report.existing) == ([folder], 0, 1)
    assert snapshot(source) == before


def test_an_unreadable_folder_is_reported_and_the_rest_import(
    tmp_path, source, raw_img_path, monkeypatch
):
    (source / "A.imp").mkdir()
    shutil.copy(raw_img_path, source / "b.jpg")
    iterdir = Path.iterdir

    def deny(self):
        if self.name == "A.imp":
            raise PermissionError("denied")
        return iterdir(self)

    monkeypatch.setattr(Path, "iterdir", deny)
    report = build(tmp_path).import_inputs([source])
    assert report.failed == ["A.imp: denied"]
    assert report.pages == [source / "b.imp"]


def test_an_unreadable_results_check_is_reported(tmp_path, source, monkeypatch):
    (source / "A.imp").mkdir()
    pipeline = build(tmp_path)

    def deny(path):
        raise PermissionError("denied")

    monkeypatch.setattr(pipeline, "_is_existing_collectra_folder", deny)
    assert pipeline.import_inputs([source]).failed == ["A.imp: denied"]


def test_folders_are_listed_once_whatever_the_number_of_files(
    tmp_path, source, raw_img_path, monkeypatch
):
    pipeline = build(tmp_path)
    calls = []
    iterdir = Path.iterdir

    def counting(self):
        calls.append(self)
        return iterdir(self)

    monkeypatch.setattr(Path, "iterdir", counting)
    for count in (2, 6):
        folder = source / f"n{count}"
        folder.mkdir()
        for number in range(count):
            shutil.copy(raw_img_path, folder / f"{number}.jpg")
        calls.clear()
        pipeline.import_inputs([folder])
        assert calls.count(folder) == 1
        assert len(calls) <= 1 + count  # import_file may list its own folder only


def test_every_input_must_exist_before_any_is_imported(tmp_path, source, raw_img_path):
    shutil.copy(raw_img_path, source / "one.jpg")
    with pytest.raises(FileNotFoundError, match="gone"):
        build(tmp_path).import_inputs([source / "one.jpg", source / "gone"])
    assert not (source / "one.imp").exists()
