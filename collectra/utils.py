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

import zipfile, yaml, os, importlib
from contextlib import contextmanager
from typing import List
from pathlib import Path
from tqdm import tqdm
from PIL import Image as ImagePil



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


def crop(
    path: Path | str, coordinates: tuple[float, float, float, float], show=False
) -> ImagePil.Image:
    """Crop an image using the specified coordinates.

    Opens an image file and crops it to the specified rectangular region
    defined by the coordinates tuple. Optionally displays the cropped image.

    Args:
        path (Path): Path to the image file to crop.
        coordinates (tuple[float, float, float, float]): Crop coordinates as
            (left, upper, right, lower) in pixels.
        show (bool, optional): Whether to display the cropped image. Defaults to False.

    Returns:
        ImagePil.Image: The cropped image object.

    Raises:
        FileNotFoundError: If the image file doesn't exist.
        PIL.UnidentifiedImageError: If the file is not a valid image format.
    """
    with ImagePil.open(path) as imf:
        im_crop = imf.crop(coordinates)
        if show:
            im_crop.show()
    return im_crop

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