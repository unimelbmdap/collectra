from pathlib import Path

import pytest
import yaml

from collectra.gui.arrangement import ArrangementStore, sanitize_arrangement

SAMPLE = {
    "version": 1,
    "focused": "bottom",
    "layout": [],
    "sizes": {},
    "panels": {
        "top": {"active": 0, "tabs": [{"kind": "dag"}]},
        "bottom": {
            "active": 1,
            "tabs": [
                {"kind": "node", "group": "specimen_sheet"},
                {"kind": "node", "group": "primary_label_unoriented"},
            ],
        },
    },
}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "collectra.gui.arrangement.user_cache_dir",
        lambda *a, **k: str(tmp_path / "cache"),
    )
    return ArrangementStore(tmp_path / "project")


def test_nothing_saved_yet_reads_as_none(store):
    assert store.read() is None


def test_named_directory_preserves_separate_pipelines(tmp_path):
    one = ArrangementStore(tmp_path / "one", pipeline_name="My Workflow")
    other = ArrangementStore(tmp_path / "two", pipeline_name="My Workflow")
    assert one.path.parent.name == "My Workflow"
    assert one.path != other.path


def test_legacy_user_layout_takes_priority_over_workflow_default(store, tmp_path):
    default = tmp_path / "default.yaml"
    default.write_text(yaml.safe_dump({**SAMPLE, "focused": "top"}))
    store.default_layout = default
    assert store.read()["focused"] == "top"
    store.legacy_path.parent.mkdir(parents=True, exist_ok=True)
    store.legacy_path.write_text(yaml.safe_dump(SAMPLE))
    assert store.read() == SAMPLE
    store.write({**SAMPLE, "focused": "new"})
    assert store.read()["focused"] == "new"


@pytest.mark.parametrize("contents", ["not a layout", ": : [", "version: 99"])
def test_bad_user_layout_falls_back_to_default(store, tmp_path, contents):
    default = tmp_path / "default.yaml"
    default.write_text(yaml.safe_dump(SAMPLE))
    store.default_layout = default
    store.path.parent.mkdir(parents=True)
    store.path.write_text(contents)
    assert store.read() == SAMPLE


def test_what_is_written_is_read_back(store):
    store.write(SAMPLE)

    assert store.read() == SAMPLE


def test_it_is_a_readable_yaml_dictionary_in_the_per_user_cache(store, tmp_path):
    store.write(SAMPLE)

    assert store.path.suffix == ".yaml"
    assert str(store.path).startswith(str(tmp_path / "cache"))
    assert yaml.safe_load(store.path.read_text())["panels"]["bottom"]["active"] == 1


def test_each_pipeline_has_its_own_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "collectra.gui.arrangement.user_cache_dir",
        lambda *a, **k: str(tmp_path / "cache"),
    )
    one, other = ArrangementStore(tmp_path / "a"), ArrangementStore(tmp_path / "b")

    one.write(SAMPLE)

    assert other.read() is None


def test_a_corrupt_file_reads_as_none(store):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(": : not [valid yaml")

    assert store.read() is None


def test_a_file_of_the_wrong_shape_reads_as_none(store):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("- just\n- a list\n")

    assert store.read() is None


def test_writing_replaces_the_previous_arrangement(store):
    store.write(SAMPLE)
    store.write(
        {"version": 1, "focused": "top", "panels": {}, "layout": [], "sizes": {}}
    )

    assert store.read()["focused"] == "top"


def test_an_unwritable_location_does_not_raise(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed")
    monkeypatch.setattr(
        "collectra.gui.arrangement.user_cache_dir", lambda *a, **k: str(blocker)
    )

    ArrangementStore(tmp_path / "project").write(SAMPLE)


class TestSanitize:
    def test_keeps_a_valid_arrangement(self):
        assert sanitize_arrangement(SAMPLE) == SAMPLE

    def test_rejects_non_dictionaries(self):
        assert sanitize_arrangement(None) is None
        assert sanitize_arrangement([1]) is None
        assert sanitize_arrangement("x") is None

    def test_drops_tabs_with_an_unknown_kind_or_no_group(self):
        data = {
            "version": 1,
            "panels": {
                "bottom": {
                    "active": 0,
                    "tabs": [
                        {"kind": "weird", "group": "a"},
                        {"kind": "node"},
                        {"kind": "node", "group": ""},
                        {"kind": "node", "group": "ok"},
                        "not a dict",
                    ],
                }
            },
        }

        assert sanitize_arrangement(data)["panels"]["bottom"]["tabs"] == [
            {"kind": "node", "group": "ok"}
        ]

    def test_clamps_the_active_index_into_range(self):
        data = {
            "version": 1,
            "panels": {"bottom": {"active": 9, "tabs": [{"kind": "dag"}]}},
        }

        assert sanitize_arrangement(data)["panels"]["bottom"]["active"] == 0

    def test_ignores_extra_keys_and_non_string_panel_names(self):
        data = {
            "version": 1,
            "extra": 1,
            "panels": {3: {"tabs": []}, "top": {"tabs": [], "junk": 1}},
        }

        result = sanitize_arrangement(data)

        assert "extra" not in result
        assert list(result["panels"]) == ["top"]
        assert "junk" not in result["panels"]["top"]

    def test_an_unknown_future_version_is_not_trusted(self):
        assert sanitize_arrangement({"version": 99, "panels": {}}) is None


class TestLayout:
    def test_keeps_the_log_of_splits_and_closes_in_order(self):
        data = {
            "version": 1,
            "panels": {},
            "layout": [
                {
                    "op": "split",
                    "from": "bottom",
                    "direction": "right",
                    "key": "panel2",
                },
                {
                    "op": "split",
                    "from": "panel2",
                    "direction": "below",
                    "key": "panel3",
                },
                {"op": "remove", "key": "panel2"},
            ],
        }

        assert sanitize_arrangement(data)["layout"] == data["layout"]

    def test_drops_malformed_or_unknown_operations(self):
        data = {
            "version": 1,
            "layout": [
                {
                    "op": "split",
                    "from": "bottom",
                    "direction": "sideways",
                    "key": "panel2",
                },
                {"op": "split", "from": "bottom", "direction": "left"},
                {"op": "explode", "key": "x"},
                {"op": "remove"},
                "nope",
                {"op": "remove", "key": "panel4"},
            ],
        }

        assert sanitize_arrangement(data)["layout"] == [
            {"op": "remove", "key": "panel4"}
        ]

    def test_the_log_is_capped_to_the_most_recent_operations(self):
        ops = [{"op": "remove", "key": f"panel{i}"} for i in range(500)]

        layout = sanitize_arrangement({"version": 1, "layout": ops})["layout"]

        assert len(layout) == 200
        assert layout[-1] == {"op": "remove", "key": "panel499"}

    def test_a_missing_log_is_an_empty_one(self):
        assert sanitize_arrangement({"version": 1})["layout"] == []

    def test_keeps_panel_sizes_that_are_pixel_flex_values(self):
        data = {
            "version": 1,
            "sizes": {
                "top": "0 0 320px",
                "bottom": "0 0 410.5px",
                "panel2": "calc(evil)",
                "panel3": 7,
                4: "0 0 10px",
            },
        }

        assert sanitize_arrangement(data)["sizes"] == {
            "top": "0 0 320px",
            "bottom": "0 0 410.5px",
        }

    def test_keeps_shrinkable_percentage_sizes(self):
        data = {
            "version": 1,
            "sizes": {"a": "0 1 45.50%", "b": "0 2 10%", "c": "1 1 30%"},
        }

        assert sanitize_arrangement(data)["sizes"] == {"a": "0 1 45.50%"}

    def test_keeps_percentage_sizes_too(self):
        data = {"version": 1, "sizes": {"top": "0 0 45.50%", "bottom": "0 0 100%x"}}

        assert sanitize_arrangement(data)["sizes"] == {"top": "0 0 45.50%"}

    def test_missing_sizes_are_an_empty_dictionary(self):
        assert sanitize_arrangement({"version": 1})["sizes"] == {}


class TestCollapsed:
    def test_keeps_the_list_of_collapsed_panels(self):
        data = {"version": 1, "collapsed": ["top", "panel2"]}

        assert sanitize_arrangement(data)["collapsed"] == ["top", "panel2"]

    def test_an_empty_list_means_none_are_collapsed(self):
        assert sanitize_arrangement({"version": 1, "collapsed": []})["collapsed"] == []

    def test_a_save_without_the_list_stays_without_it(self):
        # An older save must not be read as "nothing is collapsed".
        assert "collapsed" not in sanitize_arrangement({"version": 1})

    def test_drops_anything_that_is_not_a_panel_name(self):
        data = {"version": 1, "collapsed": ["top", 7, None, "", "panel3"]}

        assert sanitize_arrangement(data)["collapsed"] == ["top", "panel3"]

    def test_a_list_of_the_wrong_kind_is_treated_as_missing(self):
        assert "collapsed" not in sanitize_arrangement(
            {"version": 1, "collapsed": "top"}
        )

    def test_round_trips_through_the_file(self, store):
        store.write(
            {
                "version": 1,
                "layout": [],
                "sizes": {},
                "panels": {},
                "collapsed": ["top"],
            }
        )

        assert store.read()["collapsed"] == ["top"]
