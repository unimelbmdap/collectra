"""The Settings dialog's view of the environment variables stored for a pipeline.

Every call returns the whole list again, so the dialog only has to redraw it.
Private values reach the page masked; the page never sees them in full.
"""

import os
from types import SimpleNamespace

import pytest
from keyring.backends import fail

from collectra.gui.backend import GUIBackend
from collectra.pipelines import env as env_module
from collectra.pipelines.base import Pipeline

PLAIN = "COLLECTRA_TEST_PLAIN"
SECRET = "COLLECTRA_TEST_SECRET"
OTHER = "COLLECTRA_TEST_OTHER"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (PLAIN, SECRET, OTHER):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def backend(tmp_path):
    return GUIBackend(Pipeline("hespi", "hp", "1", path=tmp_path))


def test_the_listing_names_the_pipeline_and_where_variables_are_kept(backend):
    listing = backend.get_env_variables()
    assert listing == {
        "success": True,
        "pipeline": "hespi",
        "path": str(env_module.PipelineEnv("hespi").path),
        "keyring_available": True,
        "variables": [],
    }


def test_variables_are_listed_with_private_values_masked(backend, monkeypatch):
    backend.set_env_variable(PLAIN, "/data/hf")
    backend.set_env_variable(SECRET, "sk-ant-api03-abcdefgh", True)
    monkeypatch.setenv(OTHER, "from the shell")
    listing = backend.set_env_variable(OTHER, "stored")
    assert listing["success"] is True
    assert listing["variables"] == [
        {
            "name": OTHER,
            "value": "stored",
            "private": False,
            "overridden": True,
            "override": False,
            "replaces_shell": False,
            "missing": False,
        },
        {
            "name": PLAIN,
            "value": "/data/hf",
            "private": False,
            "overridden": False,
            "override": False,
            "replaces_shell": False,
            "missing": False,
        },
        {
            "name": SECRET,
            "value": "sk-a…efgh",
            "private": True,
            "overridden": False,
            "override": False,
            "replaces_shell": False,
            "missing": False,
        },
    ]
    assert "abcd" not in str(listing)


def test_a_private_value_missing_from_the_keyring_is_marked(
    backend, pipeline_env_keyring
):
    backend.set_env_variable(SECRET, "secret", True)
    pipeline_env_keyring.delete_password("collectra:hespi", SECRET)
    (variable,) = backend.get_env_variables()["variables"]
    assert variable["missing"] is True and variable["value"] == ""


def test_a_value_set_in_the_dialog_reaches_runs_started_from_the_gui(backend):
    backend.set_env_variable(PLAIN, "/data/hf")
    assert os.environ[PLAIN] == "/data/hf"
    backend.delete_env_variable(PLAIN)
    assert PLAIN not in os.environ


def test_deleting_and_clearing(backend):
    backend.set_env_variable(PLAIN, "a")
    backend.set_env_variable(OTHER, "b")
    listing = backend.delete_env_variable(PLAIN)
    assert [variable["name"] for variable in listing["variables"]] == [OTHER]
    listing = backend.clear_env_variables()
    assert listing["success"] is True and listing["variables"] == []


def test_problems_are_reported_with_the_listing(backend, monkeypatch):
    backend.set_env_variable(PLAIN, "kept")
    result = backend.set_env_variable("MY-KEY", "value")
    assert result["success"] is False
    assert "not a valid environment variable name" in result["error"]
    assert [variable["name"] for variable in result["variables"]] == [PLAIN]

    result = backend.delete_env_variable(OTHER)
    assert result["success"] is False and "not stored" in result["error"]

    monkeypatch.setattr(env_module, "keyring_backend", fail.Keyring)
    result = backend.set_env_variable(SECRET, "secret", True)
    assert result["success"] is False and "No secure keyring" in result["error"]
    assert result["keyring_available"] is False


def test_a_damaged_file_is_reported_rather_than_replaced(backend):
    path = env_module.PipelineEnv("hespi").path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("variables: [unclosed\n")
    for result in (
        backend.get_env_variables(),
        backend.set_env_variable(PLAIN, "value"),
        backend.clear_env_variables(),
    ):
        assert result["success"] is False
        assert "hespi.yaml" in result["error"]
        assert result["variables"] == []
    assert path.read_text() == "variables: [unclosed\n"


def test_a_pipeline_without_stored_variables_says_so():
    result = GUIBackend(SimpleNamespace(ext="collectra")).get_env_variables()
    assert result == {
        "success": False,
        "error": "This pipeline has no settings.",
        "variables": [],
    }


def test_a_variable_can_override_the_shell(backend, monkeypatch):
    monkeypatch.setenv(PLAIN, "from the shell")
    listing = backend.set_env_variable(PLAIN, "stored", False, True)
    (variable,) = listing["variables"]
    assert variable["override"] and variable["replaces_shell"]
    assert not variable["overridden"]
    assert os.environ[PLAIN] == "stored"

    (variable,) = backend.set_env_override(PLAIN, False)["variables"]
    assert variable["overridden"] and not variable["override"]
    assert os.environ[PLAIN] == "from the shell"

    result = backend.set_env_override(OTHER, True)
    assert result["success"] is False and "not stored" in result["error"]
