"""Results files are replaced in one step, so an interrupted write loses nothing."""

import pytest
import yaml

from collectra import utils
from collectra.utils import write_yaml


def test_write_yaml_writes_each_entry_in_order(tmp_path):
    path = tmp_path / "results.yaml"
    write_yaml({"b": 1, "a": [1, 2]}, path)
    assert list(yaml.safe_load(path.read_text())) == ["b", "a"]
    assert [p.name for p in tmp_path.iterdir()] == ["results.yaml"]


def test_an_interrupted_write_leaves_the_previous_file_whole(tmp_path, monkeypatch):
    path = tmp_path / "results.yaml"
    write_yaml({"page": "a.jpg", "found": [1, 2, 3]}, path)
    before = path.read_text()
    calls = []

    def dump(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise KeyboardInterrupt  # e.g. the job is stopped mid-write
        return real(*args, **kwargs)

    real = yaml.dump
    monkeypatch.setattr(utils.yaml, "dump", dump)
    with pytest.raises(KeyboardInterrupt):
        write_yaml({"page": "a.jpg", "found": [4, 5, 6, 7]}, path)
    assert path.read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["results.yaml"]
