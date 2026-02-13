"""
Fixtures for converter tests — provides a realistic YOLO dataset directory structure.
"""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def yolo_dataset(tmp_path) -> tuple[Path, Path]:
    """Create a minimal YOLO dataset directory structure for testing.

    Structure:
        dataset/
            config.yaml
            train.txt
            val.txt
            images/
                img_0.jpg
                img_1.jpg
                img_2.jpg
                img_3.jpg
            labels/
                img_0.txt
                img_1.txt
                img_2.txt
                img_3.txt

    Returns:
        tuple[Path, Path]: (path to config.yaml, path to dataset dir)
    """
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    images_dir = dataset_dir / "images"
    images_dir.mkdir()
    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir()

    class_names = ["cat", "dog"]

    # Create image files (small dummy content)
    for i in range(4):
        img = images_dir / f"img_{i}.jpg"
        # Write minimal JPEG-like content (just needs to exist for path checks)
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        # Create corresponding YOLO annotation files
        txt = labels_dir / f"img_{i}.txt"
        txt.write_text(
            f"0 0.5 0.5 0.1 0.2\n"  # class 0 (cat)
            f"1 0.3 0.7 0.05 0.1\n"  # class 1 (dog)
        )

    # Create train/val file lists (relative paths)
    train_files = [f"images/img_{i}.jpg" for i in range(3)]
    val_files = [f"images/img_3.jpg"]

    train_txt = dataset_dir / "train.txt"
    train_txt.write_text("\n".join(train_files) + "\n")

    val_txt = dataset_dir / "val.txt"
    val_txt.write_text("\n".join(val_files) + "\n")

    # Create YOLO config YAML
    config_data = {
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(class_names),
        "names": class_names,
    }
    config_path = dataset_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path, dataset_dir
