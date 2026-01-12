"""
Ensemble module for collectra project.

This module provides functionality to ensemble multiple collectra processing results
from different sources (e.g., different LLM models) into a unified output structure.

The module handles:
- Finding collectra files (folders with specific extensions like .grapto)
- Verifying files exist across all input folders
- Creating ensemble output with merged results
- Generating link.yaml to track source mappings
"""

import logging
import shutil
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def find_collectra_files(folder: Path, extension: str = ".grapto") -> dict[str, Path]:
    """
    Find all collectra files (folders ending with the specified extension) in a folder.

    Args:
        folder: The folder to search for collectra files.
        extension: The file extension to look for (default: ".grapto").

    Returns:
        A dictionary mapping filename (str) to its full path (Path).

    Raises:
        FileNotFoundError: If the folder does not exist.
        PermissionError: If the folder cannot be accessed.
        ValueError: If multiple files with the same name are found in the folder.

    Example:
        >>> files = find_collectra_files(Path("/data/results"), ".grapto")
        >>> print(files)
        {'MMRIRN1505070_P350015.grapto': Path('/data/results/MMRIRN1505070_P350015.grapto')}
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    collectra_files: dict[str, Path] = {}
    duplicates: list[str] = []

    try:
        for item in folder.iterdir():
            if item.is_dir() and item.name.endswith(extension):
                filename = item.name
                if filename in collectra_files:
                    duplicates.append(filename)
                else:
                    collectra_files[filename] = item
    except PermissionError as e:
        raise PermissionError(f"Cannot access folder: {folder}") from e

    if duplicates:
        raise ValueError(
            f"Multiple instances of same filename found in folder '{folder}': {duplicates}"
        )

    return collectra_files


def verify_collectra_files(
    folders: list[Path], extension: str = ".grapto"
) -> tuple[dict[str, list[Path]], list[str]]:
    """
    Verify that collectra files exist in all input folders.

    This function finds all collectra files across all input folders and identifies
    which files exist in ALL folders (verified) vs those that are missing from some
    folders (warnings).

    Args:
        folders: List of folders to search for collectra files.
        extension: The file extension to look for (default: ".grapto").

    Returns:
        A tuple containing:
        - dict[str, list[Path]]: Mapping of verified filenames to their paths in all folders
          (in the same order as the input folders list).
        - list[str]: List of warning messages for files not found in all folders.

    Raises:
        ValueError: If no folders are provided or if duplicate files are found in any folder.
        FileNotFoundError: If any folder does not exist.

    Example:
        >>> folders = [Path("/data/model1"), Path("/data/model2")]
        >>> verified, warnings = verify_collectra_files(folders, ".grapto")
        >>> print(verified)
        {'file1.grapto': [Path('/data/model1/file1.grapto'), Path('/data/model2/file1.grapto')]}
    """
    if not folders:
        raise ValueError("At least one input folder must be provided")

    # Find all collectra files in each folder
    folder_files: list[dict[str, Path]] = []
    for folder in folders:
        files = find_collectra_files(folder, extension)
        folder_files.append(files)

    # Get all unique filenames across all folders
    all_filenames: set[str] = set()
    for files in folder_files:
        all_filenames.update(files.keys())

    verified_files: dict[str, list[Path]] = {}
    warnings: list[str] = []

    for filename in sorted(all_filenames):
        paths: list[Path] = []
        missing_folders: list[Path] = []

        for folder, files in zip(folders, folder_files):
            if filename in files:
                paths.append(files[filename])
            else:
                missing_folders.append(folder)

        if missing_folders:
            # File not found in all folders - add warning
            missing_str = ", ".join(str(f) for f in missing_folders)
            warning_msg = f"File '{filename}' not found in folders: {missing_str}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        else:
            # File found in all folders - verified
            verified_files[filename] = paths

    return verified_files, warnings


def _get_image_file(collectra_folder: Path) -> Path | None:
    """
    Get the image file from a collectra folder.

    The image file is any file that is NOT results.yaml or usage.yaml.

    Args:
        collectra_folder: Path to the collectra folder.

    Returns:
        Path to the image file, or None if no image file is found.
    """
    excluded_files = {"results.yaml", "usage.yaml"}

    try:
        for item in collectra_folder.iterdir():
            if item.is_file() and item.name not in excluded_files:
                return item
    except PermissionError:
        logger.warning(f"Cannot access folder to find image: {collectra_folder}")
        return None

    return None


def create_ensemble_output(
    verified_files: dict[str, list[Path]],
    output_folder: Path,
    extension: str = ".grapto",
) -> Path:
    """
    Create the ensemble output folder structure.

    For each verified file, this function:
    - Creates a folder with the same name in the output directory
    - Copies the image file from the first source folder
    - Creates an empty results.yaml file
    - Generates a link.yaml file mapping each output to its source folders

    Args:
        verified_files: Dictionary mapping filenames to their paths in source folders.
        output_folder: The output folder where ensemble results will be created.
        extension: The file extension being used (default: ".grapto").

    Returns:
        Path to the created link.yaml file.

    Raises:
        FileExistsError: If the output folder already exists and is not empty.
        PermissionError: If the output folder cannot be created or written to.
        ValueError: If verified_files is empty.

    Example:
        >>> verified = {'file1.grapto': [Path('/src1/file1.grapto'), Path('/src2/file1.grapto')]}
        >>> link_yaml = create_ensemble_output(verified, Path('/output'), ".grapto")
        >>> print(link_yaml)
        Path('/output/link.yaml')
    """
    if not verified_files:
        raise ValueError("No verified files to process")

    # Create output folder
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot create output folder: {output_folder}") from e

    # Prepare link.yaml content
    link_data: dict[str, list[str]] = {}

    for filename, source_paths in verified_files.items():
        # Create the output collectra folder
        output_collectra_folder = output_folder / filename

        try:
            output_collectra_folder.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot create folder: {output_collectra_folder}"
            ) from e

        # Create empty results.yaml
        results_yaml_path = output_collectra_folder / "results.yaml"
        try:
            results_yaml_path.touch()
        except PermissionError as e:
            raise PermissionError(
                f"Cannot create results.yaml: {results_yaml_path}"
            ) from e

        # Copy image from first source folder
        first_source = source_paths[0]
        image_file = _get_image_file(first_source)

        if image_file:
            dest_image = output_collectra_folder / image_file.name
            try:
                shutil.copy2(image_file, dest_image)
            except PermissionError as e:
                raise PermissionError(
                    f"Cannot copy image from {image_file} to {dest_image}"
                ) from e
            except FileNotFoundError:
                logger.warning(f"Image file not found: {image_file}")
        else:
            logger.warning(f"No image file found in source folder: {first_source}")

        # Add to link data
        link_data[filename] = [str(p) for p in source_paths]

    # Write link.yaml
    link_yaml_path = output_folder / "link.yaml"
    try:
        with open(link_yaml_path, "w") as f:
            yaml.dump(link_data, f, default_flow_style=False, sort_keys=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot write link.yaml: {link_yaml_path}") from e

    logger.info(f"Created ensemble output at: {output_folder}")
    logger.info(f"Processed {len(verified_files)} files")
    logger.info(f"Link file created at: {link_yaml_path}")

    return link_yaml_path


def ensemble_files(
    input_folders: list[Path],
    output_folder: Path,
    extension: str = ".grapto",
) -> Path:
    """
    Main entry point for ensembling collectra files from multiple source folders.

    This function combines the functionality of finding, verifying, and creating
    ensemble output for collectra files across multiple input folders.

    The process:
    1. Finds all collectra files in each input folder
    2. Verifies which files exist in ALL input folders
    3. Logs warnings for files missing from some folders
    4. Creates output structure with:
       - A folder for each verified file
       - Empty results.yaml in each folder
       - Copy of the image from the first source
       - link.yaml mapping each output to its sources

    Args:
        input_folders: List of folders containing collectra files to ensemble.
        output_folder: The output folder where ensemble results will be created.
        extension: The file extension to look for (default: ".grapto").

    Returns:
        Path to the created link.yaml file.

    Raises:
        ValueError: If no input folders provided, no files found, or no verified files.
        FileNotFoundError: If any input folder does not exist.
        PermissionError: If folders cannot be accessed or written to.

    Example:
        >>> input_folders = [Path("/data/model1"), Path("/data/model2")]
        >>> output = Path("/data/ensemble")
        >>> link_yaml = ensemble_files(input_folders, output)
        >>> print(link_yaml)
        Path('/data/ensemble/link.yaml')
    """
    if not input_folders:
        raise ValueError("At least one input folder must be provided")

    # Convert to Path objects if needed
    input_folders = [Path(f) if not isinstance(f, Path) else f for f in input_folders]
    output_folder = (
        Path(output_folder) if not isinstance(output_folder, Path) else output_folder
    )

    logger.info(f"Starting ensemble process with {len(input_folders)} input folders")
    logger.info(f"Input folders: {[str(f) for f in input_folders]}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Extension: {extension}")

    # Verify collectra files across all folders
    verified_files, warnings = verify_collectra_files(input_folders, extension)

    if warnings:
        logger.warning(f"Found {len(warnings)} files not present in all folders:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    if not verified_files:
        raise ValueError(
            "No files found that exist in all input folders. "
            "Check warnings for details on missing files."
        )

    logger.info(
        f"Found {len(verified_files)} files present in all {len(input_folders)} folders"
    )

    # Create ensemble output
    link_yaml_path = create_ensemble_output(verified_files, output_folder, extension)

    logger.info("Ensemble process completed successfully")

    return link_yaml_path


def load_link_yaml(link_yaml_path: Path) -> dict[str, list[Path]]:
    """
    Load and parse a link.yaml file.

    Args:
        link_yaml_path: Path to the link.yaml file.

    Returns:
        Dictionary mapping filenames to their source paths.

    Raises:
        FileNotFoundError: If the link.yaml file does not exist.
        yaml.YAMLError: If the file cannot be parsed as YAML.

    Example:
        >>> links = load_link_yaml(Path("/output/link.yaml"))
        >>> print(links)
        {'file1.grapto': [Path('/src1/file1.grapto'), Path('/src2/file1.grapto')]}
    """
    if not link_yaml_path.exists():
        raise FileNotFoundError(f"Link file not found: {link_yaml_path}")

    with open(link_yaml_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    # Convert string paths to Path objects
    result: dict[str, list[Path]] = {}
    for filename, paths in data.items():
        result[filename] = [Path(p) for p in paths]

    return result


def get_ensemble_folder(link_yaml_path: Path) -> Path:
    """
    Get the ensemble output folder from a link.yaml path.

    Args:
        link_yaml_path: Path to the link.yaml file.

    Returns:
        Path to the ensemble output folder (parent of link.yaml).

    Example:
        >>> folder = get_ensemble_folder(Path("/output/link.yaml"))
        >>> print(folder)
        Path('/output')
    """
    return link_yaml_path.parent
