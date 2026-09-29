import yaml

from collectra.tasks.machine_learning.training import _limit_training_files


def make_item(folder, name, partition):
    item = folder / f"{name}.palynomorph"
    item.mkdir()
    metadata = {"partition": partition} if partition else {}
    (item / "results.yaml").write_text(
        yaml.safe_dump({"collectra_results_metadata": metadata})
    )
    return item


def test_limit_training_files_keeps_max_items_per_split(tmp_path):
    files = [make_item(tmp_path, f"t{i}", "") for i in range(5)]
    files += [make_item(tmp_path, f"v{i}", "validation") for i in range(4)]
    files += [make_item(tmp_path, f"x{i}", "exclude") for i in range(3)]

    selected = _limit_training_files(
        files, 2, validation="validation", exclude="exclude"
    )

    assert [item.stem for item in selected] == ["t0", "t1", "v0", "v1"]


def test_limit_training_files_without_validation_flag(tmp_path):
    files = [make_item(tmp_path, f"t{i}", "") for i in range(3)]
    files.append(make_item(tmp_path, "v0", "validation"))

    selected = _limit_training_files(files, 2)

    assert [item.stem for item in selected] == ["t0", "t1"]
