"""Persist folder sessions across backend instances without touching user caches."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from collectra.gui.backend import GUIBackend
from collectra.gui.session_cache import FolderSessionCache


def pipeline(tmp_path, name="pipeline"):
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    return SimpleNamespace(path=directory, ext="collectra")


def folder(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "results.yaml").write_text(
        "text: {type: collectra.Text, id: text, data: hello}"
    )
    return directory


def test_explicit_inputs_saved_and_restored_in_order(tmp_path):
    config = pipeline(tmp_path)
    first = folder(tmp_path, "first.collectra")
    second = folder(tmp_path, "second.collectra")
    GUIBackend(config, inputs=[second, first, second])
    restored = GUIBackend(config).get_initial_items()
    assert restored["provided"]
    assert [entry["path"] for entry in restored["folders"]] == [str(second), str(first)]
    assert (
        GUIBackend(pipeline(tmp_path, "other")).get_initial_items()["provided"] is False
    )
    GUIBackend(config, inputs=[first])
    assert [
        entry["path"] for entry in GUIBackend(config).get_initial_items()["folders"]
    ] == [str(first)]


def test_dialog_selection_replaces_session(tmp_path):
    config = pipeline(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    first = folder(parent, "a.collectra")
    second = folder(parent, "b.collectra")
    api = GUIBackend(config)
    window = MagicMock()
    window.create_file_dialog.return_value = [str(parent)]
    api.set_window(window)
    assert api.select_collectra_path()["success"]
    assert [
        entry["path"] for entry in GUIBackend(config).get_initial_items()["folders"]
    ] == [str(first), str(second)]
    window.create_file_dialog.return_value = [str(second)]
    assert api.select_collectra_path()["success"]
    assert [
        entry["path"] for entry in GUIBackend(config).get_initial_items()["folders"]
    ] == [str(second)]
    window.create_file_dialog.return_value = None
    api.select_collectra_path()
    assert [
        entry["path"] for entry in GUIBackend(config).get_initial_items()["folders"]
    ] == [str(second)]


def test_restore_skips_missing_folders_and_bad_cache(tmp_path):
    config = pipeline(tmp_path)
    valid = folder(tmp_path, "valid.collectra")
    missing_results = tmp_path / "empty.collectra"
    missing_results.mkdir()
    cache = FolderSessionCache(config.path)
    cache.write(
        [str(tmp_path / "deleted"), str(valid), str(missing_results), str(valid)]
    )
    result = GUIBackend(config).get_initial_items()
    assert [entry["path"] for entry in result["folders"]] == [str(valid)]
    assert cache.read() == [str(valid)]
    for contents in ["broken json", '{"folders": [123]}', "[]"]:
        cache.path.write_text(contents)
        assert GUIBackend(config).get_initial_items()["provided"] is False


def test_direct_yaml_load_persists_single_folder(tmp_path):
    config = pipeline(tmp_path)
    results = folder(tmp_path, "single.collectra")
    api = GUIBackend(config)
    assert api.load_yaml(str(results / "results.yaml"))["success"]
    assert GUIBackend(config).get_initial_items()["folders"][0]["path"] == str(results)


def test_unwritable_cache_does_not_block_opening(tmp_path):
    config = pipeline(tmp_path)
    results = folder(tmp_path, "single.collectra")
    api = GUIBackend(config)
    # A file in place of the cache directory produces a real write error.
    api._session_cache.path.parent.mkdir(parents=True, exist_ok=True)
    blocker = api._session_cache.path.parent / "blocker"
    blocker.write_text("not a directory")
    api._session_cache.path = blocker / "session.json"
    assert api.load_yaml(str(results / "results.yaml"))["success"]
