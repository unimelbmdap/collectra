"""Distance matrices of image embeddings, and choosing diverse images from them."""

import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collectra.utils import farthest_first, get_distance_matrix, search

# Four images on a line: a and b close together, c and d far out.
POINTS = {"a.jpg": 0.0, "b.jpg": 1.0, "c.jpg": 10.0, "d.jpg": 20.0}


@pytest.fixture(autouse=True)
def without_h5py(monkeypatch):
    """None of this needs h5py, which is no longer a dependency."""
    monkeypatch.setitem(sys.modules, "h5py", None)


@pytest.fixture
def matrix(tmp_path):
    embeddings = tmp_path / "embeddings.parquet"
    pd.DataFrame(
        {
            "filenames": list(POINTS),
            "vgg19": [np.array([x, 0.0]) for x in POINTS.values()],
        }
    ).to_parquet(embeddings)
    output = tmp_path / "out" / "distances.npz"
    get_distance_matrix(str(embeddings), str(output))
    return output


def test_the_matrix_holds_every_distance_and_the_file_names(matrix):
    with np.load(matrix) as data:
        assert data["index"].tolist() == list(POINTS)
        assert data["distmatrix"][0].tolist() == [0.0, 1.0, 10.0, 20.0]
        assert data["distmatrix"][3, 2] == 10.0


def test_the_matrix_is_saved_at_the_path_given(tmp_path, matrix):
    """``np.savez`` would add ``.npz`` to a path without it."""
    embeddings = tmp_path / "embeddings.parquet"
    output = tmp_path / "distances.matrix"
    get_distance_matrix(str(embeddings), str(output))
    assert output.is_file()
    assert not output.with_suffix(".matrix.npz").exists()


def test_the_farthest_image_from_those_chosen_comes_next(matrix):
    assert farthest_first(matrix, selected_ids=["a.jpg"], k=2) == ["d.jpg", "c.jpg"]


def test_without_a_starting_choice_one_is_picked_at_random_repeatably(matrix):
    first = farthest_first(matrix, k=2)
    assert len(first) == 2 and first == farthest_first(matrix, k=2)


def test_choosing_within_a_set_only_picks_from_that_set(matrix):
    chosen = farthest_first(
        matrix, selected_ids=["a.jpg", "b.jpg", "c.jpg"], k=2, within=True
    )
    assert set(chosen) <= {"a.jpg", "b.jpg", "c.jpg"}


def test_search_writes_the_new_choices(tmp_path, matrix):
    existing = tmp_path / "existing.txt"
    existing.write_text("a.jpg\n")
    output = tmp_path / "chosen.txt"
    assert search(str(matrix), str(existing), k=1, output=str(output)) == ["d.jpg"]
    assert output.read_text() == "d.jpg"


def test_a_matrix_saved_by_the_old_version_is_explained(tmp_path):
    old = tmp_path / "distances.h5"
    old.write_bytes(b"\x89HDF\r\n\x1a\n" + bytes(100))
    with pytest.raises(ValueError, match="older version .* get_distance_matrix"):
        farthest_first(old, k=1)


def test_h5py_is_not_a_dependency():
    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())[
        "project"
    ]
    requirements = project["dependencies"] + [
        r for extra in project["optional-dependencies"].values() for r in extra
    ]
    assert not [r for r in requirements if r.startswith("h5py")]


def test_a_matrix_that_cannot_be_made_is_reported(tmp_path, capsys):
    get_distance_matrix(str(tmp_path / "missing.parquet"), str(tmp_path / "d.npz"))
    assert "Error computing distance matrix" in capsys.readouterr().out


def test_search_needs_an_existing_matrix(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        search(str(tmp_path / "missing.npz"))


def test_search_can_add_the_new_choices_to_the_old(matrix):
    assert sorted(search(str(matrix), ["a.jpg"], k=1, new=False)) == ["a.jpg", "d.jpg"]


def test_search_within_a_set_may_return_ones_already_chosen(matrix):
    chosen = search(str(matrix), ["a.jpg", "d.jpg"], k=2, within=True)
    assert sorted(chosen) == ["a.jpg", "d.jpg"]
