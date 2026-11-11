import os
from pathlib import Path

import pytest

# Disable experiment tracking before test collection imports any ML backends.
# Subprocesses inherit these settings, including CLI and DataLoader tests.
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

pytest_plugins = [
    "tests.fixtures.functions",
    "tests.fixtures.paths",
    "tests.fixtures.models",
    "tests.fixtures.data",
    "tests.fixtures.types",
    "tests.fixtures.ensemble",
    "tests.fixtures.converters",
]


@pytest.fixture(autouse=True)
def cleanup_pt(root_data_path):
    file_pt = Path.cwd() / "yolo11n.pt"
    collectra_pt = root_data_path / "yolo11n.pt"
    if file_pt.exists():
        os.remove(file_pt)
    if collectra_pt.exists():
        os.remove(collectra_pt)


class MemoryKeyring:
    """A keyring that keeps passwords in a dict, so tests never use the real one."""

    def __init__(self):
        self.passwords = {}

    def get_password(self, service, username):
        return self.passwords.get((service, username))

    def set_password(self, service, username, password):
        self.passwords[(service, username)] = password

    def delete_password(self, service, username):
        from keyring.errors import PasswordDeleteError

        if self.passwords.pop((service, username), None) is None:
            raise PasswordDeleteError(username)


@pytest.fixture(autouse=True)
def pipeline_env_directory(tmp_path_factory, monkeypatch):
    """Keep each test's stored pipeline variables out of the user's config folder."""
    from collectra.pipelines import env

    directory = tmp_path_factory.mktemp("pipeline-env") / "env"
    monkeypatch.setattr(env, "env_directory", lambda: directory)
    return directory


@pytest.fixture(autouse=True)
def pipeline_env_keyring(monkeypatch):
    """Keep each test's private pipeline variables out of the real keyring."""
    from collectra.pipelines import env

    keyring = MemoryKeyring()
    monkeypatch.setattr(env, "keyring_backend", lambda: keyring)
    return keyring
