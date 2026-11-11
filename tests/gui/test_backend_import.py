"""Importing the raw inputs a user opens from the GUI backend."""

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from PIL import Image

from collectra.gui.backend import GUIBackend, _RunState
from collectra.pipelines.importing import ImportReport
from tests.unit.test_import_inputs import build

NOTICE = {"summary": "Imported 1 page", "failed": []}


class FakePipeline:
    """A pipeline whose import_inputs records its inputs and makes given pages."""

    name = "demo"
    version = "1.2"
    ext = "collectra"

    def __init__(self, report=None, pages=(), error=None):
        self.report = report or ImportReport()
        self.pages = pages
        self.error = error
        self.imports = []

    def import_inputs(self, inputs):
        self.imports.append(list(inputs))
        if self.error:
            raise self.error
        for page in self.pages:
            make_page(page)
        return self.report


def make_page(folder: Path) -> Path:
    folder.mkdir(exist_ok=True)
    Image.new("RGB", (40, 30)).save(folder / "page.png")
    (folder / "results.yaml").write_text(
        yaml.safe_dump(
            {"page": {"type": "collectra.Image", "id": "page", "data": "page.png"}}
        )
    )
    return folder


def picking(backend, path):
    window = MagicMock()
    window.create_file_dialog.return_value = [str(path)]
    backend.set_window(window)
    return backend


def snapshot(folder: Path) -> dict:
    return {
        path: (None if path.is_dir() else path.read_bytes(), path.stat().st_mtime_ns)
        for path in [folder, *folder.rglob("*")]
    }


@pytest.fixture
def parent(tmp_path):
    folder = tmp_path / "scans"
    folder.mkdir()
    Image.new("RGB", (40, 30)).save(folder / "page.png")
    return folder


def imported_one(parent):
    return FakePipeline(ImportReport(imported=1), pages=[parent / "page.collectra"])


def test_a_picked_parent_folder_imports_then_lists(parent):
    pipeline = imported_one(parent)
    result = picking(GUIBackend(pipeline), parent).select_collectra_path()
    assert pipeline.imports == [[parent]]
    assert result["mode"] == "parent"
    assert [folder["name"] for folder in result["folders"]] == ["page.collectra"]
    assert result["import"] == NOTICE


def test_the_parent_folder_picker_imports_too(parent):
    pipeline = imported_one(parent)
    result = picking(GUIBackend(pipeline), parent).select_parent_folder()
    assert pipeline.imports == [[parent]]
    assert [folder["name"] for folder in result["folders"]] == ["page.collectra"]
    assert result["import"] == NOTICE


def test_the_notice_names_what_was_skipped_and_failed(parent):
    report = ImportReport(
        existing=1,
        skipped=["notes.txt: not accepted"],
        skipped_folders=["empty.collectra: 0 images"],
        failed=["locked.png: denied"],
    )
    backend = GUIBackend(FakePipeline(report))
    assert backend._import_paths([parent]) == {
        "summary": report.summary("demo"),
        "failed": ["locked.png: denied"],
    }


def test_a_picked_page_folder_without_results_is_adopted(tmp_path):
    folder = tmp_path / "page.collectra"
    folder.mkdir()
    pipeline = FakePipeline(ImportReport(imported=1), pages=[folder])
    result = picking(GUIBackend(pipeline), folder).select_collectra_path()
    assert pipeline.imports == [[folder]]
    assert result["mode"] == "single"
    assert result["yaml_path"] == str(folder / "results.yaml")
    assert result["import"] == NOTICE


def test_a_page_folder_that_cannot_be_adopted_keeps_the_error(tmp_path):
    folder = tmp_path / "page.collectra"
    folder.mkdir()
    pipeline = FakePipeline(ImportReport(skipped_folders=["page.collectra: 0"]))
    result = picking(GUIBackend(pipeline), folder).select_collectra_path()
    assert result == {
        "success": False,
        "error": "No YAML or image file found in folder",
    }


def test_the_single_folder_picker_adopts_a_page_folder(tmp_path):
    folder = tmp_path / "page.collectra"
    folder.mkdir()
    pipeline = FakePipeline(ImportReport(imported=1), pages=[folder])
    result = picking(GUIBackend(pipeline), folder).select_folder()
    assert pipeline.imports == [[folder]]
    assert result["yaml_path"] == str(folder / "results.yaml")
    assert result["import"] == NOTICE


def test_the_single_folder_picker_does_not_import_other_folders(parent):
    make_page(parent)
    pipeline = imported_one(parent)
    result = picking(GUIBackend(pipeline), parent).select_folder()
    assert pipeline.imports == []
    assert result["success"] and "import" not in result


def test_opening_existing_pages_has_no_notice(tmp_path):
    page = make_page(tmp_path / "page.collectra")
    pipeline = FakePipeline(ImportReport(pages=[page], existing=1))
    backend = picking(GUIBackend(pipeline), tmp_path)
    assert "import" not in backend.select_collectra_path()
    assert "import" not in picking(backend, page).select_collectra_path()


def test_a_pipeline_without_import_opens_results_as_before(tmp_path):
    make_page(tmp_path / "page.collectra")
    backend = picking(GUIBackend(SimpleNamespace(ext="collectra")), tmp_path)
    result = backend.select_collectra_path()
    assert result["success"] and "import" not in result


@pytest.mark.parametrize(
    "error", [OSError("Permission denied"), FileNotFoundError("Path gone")]
)
def test_an_import_error_comes_back_as_the_route_error(parent, error):
    backend = picking(GUIBackend(FakePipeline(error=error)), parent)
    assert backend.select_collectra_path() == {"success": False, "error": str(error)}
    assert backend.select_parent_folder() == {"success": False, "error": str(error)}


def test_importing_while_a_run_is_active_is_allowed(tmp_path, parent):
    running = make_page(tmp_path / "running.collectra")
    backend = picking(GUIBackend(imported_one(parent)), parent)
    backend._run_state = _RunState(running.resolve(), total=1)
    with backend._run_lock:
        result = backend.select_collectra_path()
    assert result["import"] == NOTICE


def test_startup_imports_a_raw_file_and_prints_the_notice_once(parent, capsys):
    page = parent / "page.collectra"
    report = ImportReport(pages=[page], imported=1)
    pipeline = FakePipeline(report, pages=[page])
    initial = GUIBackend(pipeline, inputs=[parent / "page.png"]).get_initial_items()
    assert pipeline.imports[0] == [parent / "page.png"]
    assert [folder["path"] for folder in initial["folders"]] == [str(page)]
    assert initial["import"] == NOTICE
    assert capsys.readouterr().out == "Imported 1 page\n"


def test_startup_imports_raw_files_in_one_call(parent, capsys):
    pages = [parent / "b.collectra", parent / "a.collectra"]
    pipeline = FakePipeline(ImportReport(pages=pages, imported=2), pages=pages)
    inputs = [parent / "b.png", parent / "a.png"]
    for path in inputs:
        path.write_bytes(b"raw")
    initial = GUIBackend(pipeline, inputs=inputs).get_initial_items()
    assert pipeline.imports == [inputs]
    assert [folder["path"] for folder in initial["folders"]] == [
        str(page) for page in pages
    ]
    assert initial["import"] == {"summary": "Imported 2 pages", "failed": []}
    assert capsys.readouterr().out == "Imported 2 pages\n"


def test_startup_imports_a_folder_then_lists_it(parent, capsys):
    pipeline = imported_one(parent)
    initial = GUIBackend(pipeline, inputs=[parent]).get_initial_items()
    assert pipeline.imports == [[parent]]
    assert [folder["name"] for folder in initial["folders"]] == ["page.collectra"]
    assert initial["import"] == NOTICE


def test_startup_opens_a_file_inside_results_without_importing(tmp_path, capsys):
    page = make_page(tmp_path / "page.collectra")
    pipeline = FakePipeline(ImportReport(imported=1))
    initial = GUIBackend(pipeline, inputs=[page / "page.png"]).get_initial_items()
    assert pipeline.imports == []
    assert [folder["path"] for folder in initial["folders"]] == [str(page)]
    assert "import" not in initial
    assert capsys.readouterr().out == ""


def test_startup_keeps_its_errors(tmp_path, parent):
    pipeline = FakePipeline(ImportReport(skipped=["page.png: not accepted"]))
    with pytest.raises(FileNotFoundError, match="GUI input does not exist"):
        GUIBackend(pipeline, inputs=[parent, tmp_path / "missing"])
    assert pipeline.imports == []
    with pytest.raises(ValueError, match="No GUI results found"):
        GUIBackend(pipeline, inputs=[parent / "page.png"])
    with pytest.raises(ValueError, match="No GUI results found"):
        GUIBackend(pipeline, inputs=[parent])


def test_a_real_import_through_the_picker(tmp_path, raw_img_path):
    parent = tmp_path / "scans"
    parent.mkdir()
    shutil.copy(raw_img_path, parent / "one.jpg")
    (parent / ".DS_Store").write_text("x")
    backend = picking(GUIBackend(build(tmp_path)), parent)
    result = backend.select_collectra_path()
    assert [folder["name"] for folder in result["folders"]] == ["one.imp"]
    assert (parent / "one.imp" / "results.yaml").is_file()
    assert result["import"]["summary"] == "Imported 1 page"
    before = snapshot(parent)
    again = backend.select_collectra_path()
    assert [folder["name"] for folder in again["folders"]] == ["one.imp"]
    assert "import" not in again
    assert snapshot(parent) == before


def test_a_real_import_at_startup(tmp_path, raw_img_path, capsys):
    shutil.copy(raw_img_path, tmp_path / "one.jpg")
    initial = GUIBackend(
        build(tmp_path), inputs=[tmp_path / "one.jpg"]
    ).get_initial_items()
    assert [folder["name"] for folder in initial["folders"]] == ["one.imp"]
    assert (tmp_path / "one.imp" / "results.yaml").is_file()
    assert capsys.readouterr().out == "Imported 1 page\n"


def test_a_picked_folder_holding_results_is_not_imported_into(tmp_path, raw_img_path):
    renamed = tmp_path / "renamed"
    renamed.mkdir()
    (renamed / "results.yaml").write_text("page: {type: collectra.Image, id: page}\n")
    shutil.copy(raw_img_path, renamed / "x.jpg")
    today = picking(GUIBackend(SimpleNamespace(ext="imp")), renamed)
    before = snapshot(renamed)
    backend = picking(GUIBackend(build(tmp_path)), renamed)
    assert backend.select_collectra_path() == {
        **today.select_collectra_path(),
        "mode": "parent",
    }
    assert backend.select_parent_folder() == today.select_parent_folder()
    assert snapshot(renamed) == before


def test_startup_adopts_a_page_folder_input(tmp_path, raw_img_path, capsys):
    folder = tmp_path / "page.imp"
    folder.mkdir()
    shutil.copy(raw_img_path, folder / "page.jpg")
    initial = GUIBackend(build(tmp_path), inputs=[folder]).get_initial_items()
    assert [entry["path"] for entry in initial["folders"]] == [str(folder.resolve())]
    assert sorted(path.name for path in folder.iterdir()) == [
        "page.jpg",
        "results.yaml",
    ]
    assert initial["import"] == NOTICE
    assert capsys.readouterr().out == "Imported 1 page\n"


def test_startup_skips_a_file_the_pipeline_does_not_accept(tmp_path, raw_img_path):
    shutil.copy(raw_img_path, tmp_path / "a.jpg")
    (tmp_path / "n.txt").write_text("notes")
    pipeline = build(tmp_path)
    inputs = [tmp_path / "a.jpg", tmp_path / "n.txt"]
    initial = GUIBackend(pipeline, inputs=inputs).get_initial_items()
    assert [entry["name"] for entry in initial["folders"]] == ["a.imp"]
    assert initial["import"] == {
        "summary": "Imported 1 page, skipped 1 file Import doesn't accept",
        "failed": [],
    }
    with pytest.raises(ValueError, match="No GUI results found"):
        GUIBackend(pipeline, inputs=[tmp_path / "n.txt"])


def test_startup_checks_every_path_before_importing(tmp_path, raw_img_path):
    shutil.copy(raw_img_path, tmp_path / "a.jpg")
    with pytest.raises(FileNotFoundError, match="GUI input does not exist"):
        GUIBackend(build(tmp_path), inputs=[tmp_path / "a.jpg", tmp_path / "gone"])
    assert not (tmp_path / "a.imp").exists()
