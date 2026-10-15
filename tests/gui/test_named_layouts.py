"""Workflow presets and named user layouts in the Customize Layout menu."""

from types import SimpleNamespace

import pytest
import yaml

from collectra.gui.backend import GUIBackend


LAYOUT = {"version": 1, "panels": {}, "layout": [], "sizes": {}}


@pytest.fixture
def api(tmp_path):
    (tmp_path / "default.yaml").write_text(yaml.safe_dump(LAYOUT))
    pipeline = SimpleNamespace(
        path=tmp_path, ext="collectra", name="Workflow",
        pipeline_metadata={"name": "Workflow", "layouts": ["default.yaml"]},
    )
    return GUIBackend(pipeline)


def test_lists_and_loads_workflow_and_custom_layouts(api):
    assert api.save_named_layout("Review", {**LAYOUT, "focused": "bottom"})["success"]
    result = api.get_layouts()
    assert result["pipeline_name"] == "Workflow"
    assert result["layouts"] == [
        {"id": "pipeline:0", "source": "pipeline", "name": "default"},
        {"id": "custom:Review.yaml", "source": "custom", "name": "Review"},
    ]
    assert api.load_layout("pipeline:0")["arrangement"] == LAYOUT
    assert api.load_layout("custom:Review.yaml")["arrangement"]["focused"] == "bottom"
    assert api._arrangement.read() == LAYOUT  # named saves do not select the preset


def test_lists_existing_yaml_files_in_user_directory(api):
    directory = api._arrangement.path.parent
    directory.mkdir(parents=True)
    (directory / "Existing.yml").write_text(yaml.safe_dump(LAYOUT))
    api.save_arrangement(LAYOUT)
    assert [item["name"] for item in api.get_layouts()["layouts"]] == ["default", "Existing"]


@pytest.mark.parametrize("name", ["", " ", "../outside", "a/b", "a\\b", "a\n", ".", ".."])
def test_rejects_unsafe_or_empty_names(api, name):
    assert not api.save_named_layout(name, LAYOUT)["success"]


def test_duplicate_name_does_not_overwrite(api):
    api.save_named_layout("Review", LAYOUT)
    assert not api.save_named_layout("Review", {**LAYOUT, "focused": "top"})["success"]
    assert api.load_layout("custom:Review.yaml")["arrangement"] == LAYOUT


def test_activation_persists_chosen_layout_and_preserves_workflow_file(api):
    api.save_named_layout("Review", {**LAYOUT, "focused": "bottom"})
    assert api.activate_layout("custom:Review.yaml")["success"]
    assert api.get_arrangement()["arrangement"]["focused"] == "bottom"
    assert api.load_layout("pipeline:0")["arrangement"] == LAYOUT
    assert api.activate_layout("pipeline:0")["success"]
    assert api.get_arrangement()["arrangement"] == LAYOUT


def test_repeated_layout_changes_do_not_duplicate_sidebar_folders(api, tmp_path):
    page = tmp_path / "Page198.collectra"
    page.mkdir()
    (page / "results.yaml").write_text("text: {type: collectra.Text, id: text, data: hello}")
    api._load_input_paths([page])
    # Simulate an old session that already contains repeated paths, too.
    api._session_cache.write([str(page)] * 5)
    for _ in range(5):
        restored = api._restore_folder_session()
        assert api.activate_layout("pipeline:0")["success"]
        assert [item["path"] for item in api._collectra_folders] == [str(page)]
        assert len(restored["folders"]) == 1
        assert api._session_cache.read() == [str(page)]


def test_invalid_missing_and_arbitrary_paths_are_refused(api):
    assert not api.load_layout("custom:../../outside.yaml")["success"]
    assert not api.load_layout("pipeline:2")["success"]
    (api._pipeline.path / "default.yaml").write_text("version: 99")
    assert not api.activate_layout("pipeline:0")["success"]


def test_reveal_uses_registered_pipeline_and_custom_paths(api, monkeypatch):
    from unittest.mock import Mock

    reveal = Mock(return_value={"success": True})
    monkeypatch.setattr(api, "reveal_in_folder", reveal)
    assert api.reveal_layout("pipeline:0")["success"]
    reveal.assert_called_with(str((api._pipeline.path / "default.yaml").resolve()))
    api.save_named_layout("Review", LAYOUT)
    assert api.reveal_layout("custom:Review.yaml")["success"]
    reveal.assert_called_with(str((api._arrangement.path.parent / "Review.yaml").resolve()))
    reveal.reset_mock()
    assert not api.reveal_layout("custom:../../outside.yaml")["success"]
    reveal.assert_not_called()


def test_failed_save_reports_error(api, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Disk full")
    monkeypatch.setattr(api._arrangement, "_write_file", fail)
    assert api.save_named_layout("Review", LAYOUT) == {"success": False, "error": "Disk full"}
    assert not api.activate_layout("pipeline:0")["success"]


def test_layout_selection_never_restores_or_scans_result_data(api, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Layout selection must not reload result data")
    monkeypatch.setattr(api, "_restore_folder_session", unexpected)
    monkeypatch.setattr(api, "_scan_collectra_folder", unexpected)
    monkeypatch.setattr(api, "_folder_list_result", unexpected)
    folders = api._collectra_folders
    assert api.load_layout("pipeline:0")["success"]
    assert api.activate_layout("pipeline:0")["success"]
    assert api._collectra_folders is folders
