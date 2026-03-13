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
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import List

import yaml
from rich import print
from tqdm import tqdm
from yaml.emitter import Emitter, ScalarAnalysis

from .logger import get_logger

logger = get_logger(__name__)


def get_env(key: str, file: str = ".env") -> str:
    from dotenv import dotenv_values

    config = dotenv_values(Path.cwd() / file)
    variable = config.get(key, "") or ""
    return variable


def traceback_error(e: Exception, message: str = "", verbose: bool = False):
    if verbose:
        logger.exception("An error occurred")
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


# Keep original function
_orig_analyze_scalar = Emitter.analyze_scalar


def analyze_scalar_allow_block(self, scalar):
    analysis = _orig_analyze_scalar(self, scalar)
    # For multi-line scalars, force block style to be allowed
    if isinstance(analysis, ScalarAnalysis) and "\n" in scalar:
        analysis.allow_block = True
    return analysis


Emitter.analyze_scalar = analyze_scalar_allow_block


class LiteralDumper(yaml.SafeDumper):
    pass


def str_presenter(dumper, data):
    if "\n" in data:
        # Force literal block scalar (|)
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


LiteralDumper.add_representer(str, str_presenter)


def write_yaml(data, filepath: Path | str):
    with open(filepath, "w") as f:
        for file_key, data_item in data.items():
            yaml.dump(
                {file_key: data_item},
                f,
                Dumper=LiteralDumper,
                sort_keys=False,
                allow_unicode=True,
            )
            f.write("\n")
