"""Utility functions and helpers for the Collectra workflow system.

This module provides common utility functions used throughout the Collectra
application, including file operations, message formatting, image processing,
and configuration management.

The module includes functions for:
    - Formatted console output messages (success, error, processing)
    - File collection and filtering operations
    - Configuration loading from directories and zip files
    - Image processing and cropping operations
    - Dynamic class path resolution

Functions:
    success_msg: Format success messages for console output
    error_msg: Format error messages for console output
    processing_msg: Format processing status messages
    get_class_path: Get fully qualified class path from object
    get_all_files: Collect files matching format from paths
    unzip: Extract configuration from zip files
    from_dir: Load configuration from directory
    load_class_from_string: Dynamically load class from string
    crop: Crop images using specified coordinates
"""

import importlib
import os
import traceback
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, List
from weakref import ref

import yaml
from rich import print
from tqdm import tqdm


def get_env(key: str, file: str = ".env") -> str:
    from dotenv import dotenv_values

    config = dotenv_values(Path.cwd() / file)
    variable = config.get(key, "") or ""
    return variable


def traceback_error(e: Exception, message: str = "", verbose: bool = False):
    if verbose:
        traceback.print_exc()
    if message:
        print(f"{message}\n")
    print(str(e))


def success_msg(message: str):
    """Format a success message with green styling for console output.

    Args:
        message (str): The success message text to format.

    Returns:
        str: Formatted message with Rich markup for green styling.
    """
    return f"[green]Success[/green]: {message}"


def error_msg(message: str):
    """Format an error message with red styling for console output.

    Args:
        message (str): The error message text to format.

    Returns:
        str: Formatted message with Rich markup for red styling.
    """
    return f"[red]Error[/red]: {message}"


def processing_msg(message: str):
    """Format a processing status message with orange styling for console output.

    Args:
        message (str): The processing message text to format.

    Returns:
        str: Formatted message with Rich markup for orange styling.
    """
    return f"[dark_orange]Processing[/dark_orange]: {message}"


def get_class_path(obj_or_class):
    """Get the full dotted module path for a class or instance.

    Extracts the fully qualified class path including module and class name,
    which can be used for dynamic class loading and serialization.

    Args:
        obj_or_class: Either a class object or an instance of a class.

    Returns:
        str: Fully qualified class path (e.g., 'package.module.ClassName').
    """
    if hasattr(obj_or_class, "__class__"):
        # It's an instance
        cls = obj_or_class.__class__
    else:
        # It's already a class
        cls = obj_or_class

    return f"{cls.__module__}.{cls.__name__}"


def get_all_files(data: list[str], format: str) -> List[Path]:
    """Collect all files matching the specified format from given paths.

    Recursively searches through directories and collects files that match
    the specified format. Displays a progress bar during collection.

    Args:
        data (List[str]): List of file paths or directory paths to search.
        format (str): File extension to filter by (without dot, e.g., 'jpg', 'png').

    Returns:
        List[Path]: List of Path objects for files matching the format.

    Raises:
        Exception: If no files are found matching the specified format.
    """
    files: list[Path] = []
    for path in tqdm(data, desc="Collecting files"):
        path = Path(path)
        if path.is_dir():
            sub_files = [Path(file) for file in path.glob(f"**/*{format}")]
            files.extend(sub_files)
        elif path.is_file() and path.suffix.replace(".", "") == format:
            files.append(Path(path))
    if len(files) == 0:
        raise Exception(f"No files found with format '{format}'")
    return files


def unzip(path: Path, config: str = "pipeline.yaml") -> dict:
    """Extract and load configuration data from a zip file.

    Opens a zip file and reads the specified configuration file (usually YAML)
    from within the archive, returning the parsed configuration data.

    Args:
        path (Path): Path to the zip file to extract from.
        config (str, optional): Name of config file within zip. Defaults to "pipeline.yaml".

    Returns:
        dict: Parsed configuration data from the config file.

    Raises:
        ValueError: If the config file is empty, invalid, or missing.
        zipfile.BadZipFile: If the zip file is corrupted or invalid.
    """
    with zipfile.ZipFile(path, "r") as zipf:
        data = yaml.safe_load(zipf.read(config))
        if not data:
            raise ValueError(f"Config file is empty or invalid: {path}")
        return data


def from_dir(path: Path, config: str = "pipeline.yaml") -> dict:
    """Load configuration data from a directory containing a config file.

    Reads the specified configuration file from a directory and returns
    the parsed YAML configuration data.

    Args:
        path (Path): Path to the directory containing the config file.
        config (str, optional): Name of config file. Defaults to "pipeline.yaml".

    Returns:
        dict: Parsed configuration data from the config file.

    Raises:
        ValueError: If the config file is empty, invalid, or missing.
        FileNotFoundError: If the config file doesn't exist in the directory.
        yaml.YAMLError: If the YAML file is malformed.
    """
    pipeline = path / config
    with open(pipeline, "r") as f:
        data = yaml.safe_load(f)
        if not data:
            raise ValueError(f"Config file is empty or invalid: {pipeline}")
        return data


def load_class_from_string(path: str | None):
    """Dynamically load a class from a string module path.

    Takes a fully qualified class path string and imports the class
    for instantiation. Used for loading task classes from configuration.

    Args:
        path (str): Fully qualified class path (e.g., 'module.submodule.ClassName').

    Returns:
        type: The loaded class object ready for instantiation.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
    """
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


@contextmanager
def change_dir(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def remove_exif(image_path: Path, save_path: Path):
    """Remove EXIF data from an image and save the cleaned image.

    Opens an image file, removes any embedded EXIF metadata, and saves
    the cleaned image to the specified path.

    Args:
        image_path (Path): Path to the original image file.
        save_path (Path): Path to save the image without EXIF data.
    """
    from PIL import Image as PILImage

    with PILImage.open(image_path) as img:
        data = list(img.get_flattened_data())
        image_no_exif = PILImage.new(img.mode, img.size)
        image_no_exif.putdata(data)
        image_no_exif.save(save_path, format=img.format if img.format else "JPEG")


def get_distance_matrix(
    path: str,
    output: str,
    model: str = "vgg19",
):
    """Compute the distance matrix for the embeddings"""
    import h5py
    import numpy as np
    import pandas as pd
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from scipy.spatial.distance import pdist, squareform

    try:
        input = Path(path)
        export = Path(output)
        export.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(input)
        embeddings = np.stack(df[model].values)  # type: ignore
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            progress.add_task("Computing distance matrix...", total=None)
            distvec = pdist(embeddings, metric="euclidean")

        indices = df.filenames.to_list()
        with h5py.File(export, "w") as f:
            f.create_dataset("distmatrix", data=squareform(distvec))
            f.create_dataset("index", data=np.array(indices, dtype="S"))
    except Exception as e:
        print(f"Error computing distance matrix: {e}")


def farthest_first(
    ref: Path, selected_ids: list = [], k: int = 400, within: bool = False
) -> list:
    """Perform farthest-first traversal to select k diverse samples."""
    import random

    import h5py
    import numpy as np
    from rich.progress import track

    with h5py.File(ref, "r") as f:
        dist_matrix = f["distmatrix"][:]
        indices: list[str] = [s.decode("utf-8") for s in f["index"][:]]  # type: ignore
    new_ids = []
    if within and selected_ids:
        iterator = [indices.index(sid) for sid in selected_ids]
        selected_ids = []
    else:
        iterator = [i for i, _ in enumerate(indices)]
    if len(selected_ids) == 0:
        random.seed(42)
        choice = random.choice(iterator)
        selected_ids = [choice]
        new_ids.append(choice)
        k = k - 1
    else:
        selected_ids = [indices.index(sid) for sid in selected_ids]

    for _ in track(range(k), description="Selecting diverse samples..."):
        ids = [id for id in iterator if id not in selected_ids]
        candidates = [(id, np.min(dist_matrix[id, selected_ids])) for id in ids]
        next_id = max(candidates, key=lambda x: x[1])[0]
        selected_ids.append(next_id)
        new_ids.append(next_id)
        k -= 1
    new_ids = [indices[sid] for sid in new_ids]
    return new_ids


def search(
    reference: str,
    existing: str | list = "",
    k: int = 100,
    output: str = "",
    new: bool = True,
    within: bool = False,
) -> list[str]:
    ref = Path(reference)
    if not ref.exists() or not ref.is_file():
        raise ValueError(f"Reference distance matrix file not found: {ref}")
    selected_ids = existing if isinstance(existing, list) else []
    if (
        isinstance(existing, str)
        and Path(existing).is_file()
        and Path(existing).suffix == ".txt"
    ):
        f = open(existing, "r")
        selected_ids = list(set([line.strip() for line in f if line.strip()]))
        f.close()
    new_ids = farthest_first(ref, selected_ids=selected_ids, k=k, within=within)
    new_ids = list(set(new_ids))
    duplicates = [dup for dup in new_ids if dup in selected_ids]
    if duplicates and not within:
        raise ValueError(f"Duplicate IDs found in selection: {duplicates}")
    if not new:
        new_ids = list(set(selected_ids + new_ids))
    if output:
        Path(output).write_text("\n".join(new_ids))
    return new_ids


def is_image(file_path: Path) -> bool:
    """Check if a file is an image based on its extension."""
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"]
    return file_path.suffix.lower() in image_extensions


def convert_files(config_path: Path | str) -> None:
    """Convert files based on the provided configuration."""
    from datetime import datetime

    import pytz

    with open(config_path) as f:
        config = yaml.safe_load(f)
    with change_dir(Path(config_path).parent):
        root_label = config.get("root_label", None)
        if not root_label:
            raise Exception("root_label is required in the conversion config.")
        format = config.get("format", None)
        if not format:
            raise Exception("format is required in the conversion config.")
        names: list[str] = config.get("names", [])
        if len(names) == 0:
            raise Exception("names list is required in the conversion config.")
        images = [
            image
            for image in Path(config.get("files", [])).rglob("*")
            if is_image(image)
        ]
        if len(images) == 0:
            raise Exception(f"No images found")
        output_dir = Path(config.get("output", "converted_images"))
        from rich.progress import track

        output_dir.mkdir(parents=True, exist_ok=True)
        train_split = config.get("train_split", None)
        train_ids: list[str] = [image.name for image in images]
        val_ids: list[str] = []
        if train_split:
            split_size = int(len(images) * (1 - train_split))
            val_ids: list[str] = search(
                config.get("reference", ""),
                existing=train_ids,
                k=split_size,
                within=True,
            )
            unexpected_ids = [val_id for val_id in val_ids if val_id not in train_ids]
            if unexpected_ids:
                raise ValueError(
                    f"Validation IDs not found in original set: {unexpected_ids}"
                )
            train_ids = [tid for tid in train_ids if tid not in val_ids]
            if set(val_ids).issubset(set(train_ids)):
                raise ValueError("Overlap found between training and validation IDs.")
        for image in track(images, description="Converting images"):
            converted_file_path = output_dir / f"{image.stem}.{format}"
            converted_file_path.mkdir(parents=True, exist_ok=True)
            remove_exif(image, converted_file_path / image.name)
            results_yaml = {
                "collectra_results_metadata": {
                    "timestamp": datetime.now(pytz.utc).isoformat(),
                    "validation": image.name in val_ids,
                },
                f"{root_label}": {"type": "collectra.Image", "data": image.name},
            }
            label_path = image.parent / f"{image.stem}.txt"
            if label_path.exists():
                with open(label_path, "r") as f:
                    bboxes = [line.strip() for line in f if line.strip()]
                for name in names:
                    results_yaml[name] = []
                for bbox in bboxes:
                    class_id, x_center, y_center, width_relative, height_relative = (
                        bbox.split(" ")
                    )
                    class_name = names[int(class_id)]
                    item_dimensions = {
                        "type": "collectra.ImageCrop",
                        "data": image.name,
                        "x_center": float(x_center),
                        "y_center": float(y_center),
                        "width_relative": float(width_relative),
                        "height_relative": float(height_relative),
                    }
                    results_yaml[class_name].append(item_dimensions)
                for key in results_yaml:
                    if (
                        key not in ["collectra_results_metadata", "specimen_sheet"]
                        and len(results_yaml[key]) == 1
                    ):
                        results_yaml[key] = results_yaml[key][0]
            with open(converted_file_path / "results.yaml", "w") as f:
                for key in results_yaml:
                    if not results_yaml[key]:
                        continue
                    f.write(
                        yaml.dump(
                            {key: results_yaml[key]},
                            default_flow_style=False,
                            sort_keys=False,
                        )
                    )
                    f.write("\n")
