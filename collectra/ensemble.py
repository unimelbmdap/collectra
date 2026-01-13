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

import editdistance
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

    # Run ensembling for each file
    total_stats = {"total_groups": 0, "standalones": 0, "ensembled": 0}
    for filename in verified_files.keys():
        ensemble_collectra_folder = output_folder / filename
        try:
            result = ensemble_groups_for_file(ensemble_collectra_folder, link_yaml_path)
            # Accumulate statistics
            stats = result["statistics"]
            total_stats["total_groups"] += stats["total_groups"]
            total_stats["standalones"] += stats["standalones"]
            total_stats["ensembled"] += stats["ensembled"]
        except Exception as e:
            logger.error(f"Failed to ensemble {filename}: {e}")

    logger.info(
        f"Ensemble statistics: {total_stats['total_groups']} total groups, "
        f"{total_stats['standalones']} standalones, "
        f"{total_stats['ensembled']} ensemble decisions"
    )
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


def get_source_collectra_files(
    ensemble_folder: Path, link_yaml_path: Path
) -> list[Path]:
    """
    Get the source collectra file paths for a given ensemble folder name.

    Given an ensemble collectra folder name (e.g., 'MMRIRN1505070_P350015.grapto')
    and a link.yaml path, return the list of source collectra file paths from
    link.yaml for that ensemble file.

    Args:
        ensemble_folder: The name or path of the ensemble collectra folder.
            Can be either just the folder name (e.g., 'MMRIRN1505070_P350015.grapto')
            or a full path.
        link_yaml_path: Path to the link.yaml file.

    Returns:
        List of source collectra file paths.

    Raises:
        FileNotFoundError: If the link.yaml file does not exist.
        KeyError: If the ensemble folder is not found in link.yaml.

    Example:
        >>> sources = get_source_collectra_files(
        ...     Path("MMRIRN1505070_P350015.grapto"),
        ...     Path("/data/ensemble/link.yaml")
        ... )
        >>> print(sources)
        [Path('data/sonnet45-0-8/MMRIRN1505070_P350015.grapto'), ...]
    """
    link_data = load_link_yaml(link_yaml_path)

    # Get the folder name (in case a full path was provided)
    folder_name = (
        ensemble_folder.name
        if isinstance(ensemble_folder, Path)
        else Path(ensemble_folder).name
    )

    if folder_name not in link_data:
        raise KeyError(
            f"Ensemble folder '{folder_name}' not found in link.yaml. "
            f"Available folders: {list(link_data.keys())[:5]}..."
        )

    logger.debug(f"Found {len(link_data[folder_name])} source files for {folder_name}")
    return link_data[folder_name]


def load_results_yaml(collectra_folder: Path) -> dict[str, Any]:
    """
    Load and parse the results.yaml file from a collectra folder.

    Args:
        collectra_folder: Path to the collectra folder containing results.yaml.

    Returns:
        The parsed YAML data as a dictionary.

    Raises:
        FileNotFoundError: If the results.yaml file does not exist.
        yaml.YAMLError: If the file cannot be parsed as YAML.

    Example:
        >>> results = load_results_yaml(Path("/data/sample.grapto"))
        >>> print(results.keys())
        dict_keys(['collectra_results_metadata', 'specimen_sheet', ...])
    """
    results_yaml_path = collectra_folder / "results.yaml"

    if not results_yaml_path.exists():
        raise FileNotFoundError(f"results.yaml not found in: {collectra_folder}")

    with open(results_yaml_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        logger.warning(f"results.yaml is empty in: {collectra_folder}")
        return {}

    logger.debug(f"Loaded results.yaml with {len(data)} keys from {collectra_folder}")
    return data


def _get_bounding_box_from_entry(entry: dict[str, Any]) -> dict[str, float] | None:
    """
    Extract bounding box information from a results.yaml entry.

    Args:
        entry: A dictionary entry from results.yaml that should contain
            bounding box fields.

    Returns:
        A dictionary with bounding box fields, or None if not all required
        fields are present.
    """
    required_fields = ["x_center", "y_center", "width_relative", "height_relative"]

    if all(field in entry for field in required_fields):
        return {
            "x_center": float(entry["x_center"]),
            "y_center": float(entry["y_center"]),
            "width_relative": float(entry["width_relative"]),
            "height_relative": float(entry["height_relative"]),
        }
    return None


def _find_parent_with_bounding_box(
    entry: dict[str, Any], results_data: dict[str, Any]
) -> dict[str, float] | None:
    """
    Traverse upwards through parents to find the last ImageCrop with a bounding box.

    For entries of type collectra.Text, this function traverses up through the
    parent chain to find the LAST parent with type collectra.ImageCrop and
    returns its bounding box.

    Args:
        entry: The current entry from results.yaml.
        results_data: The full results.yaml data for looking up parent entries.

    Returns:
        The bounding box dict from the last ImageCrop parent, or None if not found.
    """
    parents = entry.get("parents")

    if parents is None:
        return None

    # Parents can be a single string or a list of strings
    # When it's a list, use the LAST parent in the list
    if isinstance(parents, list):
        if not parents:
            return None
        parent_id = parents[-1]  # Get the last parent
    else:
        parent_id = parents  # Single parent as string

    # Find the parent entry by matching the id field
    parent_entry = None
    for key, value in results_data.items():
        if isinstance(value, dict) and value.get("id") == parent_id:
            parent_entry = value
            break

    if parent_entry is None:
        logger.debug(f"Parent with id '{parent_id}' not found in results.yaml")
        return None

    # Check if parent is an ImageCrop
    if parent_entry.get("type") == "collectra.ImageCrop":
        bounding_box = _get_bounding_box_from_entry(parent_entry)
        if bounding_box:
            return bounding_box

    # If not an ImageCrop, continue traversing upward
    return _find_parent_with_bounding_box(parent_entry, results_data)


def extract_labels_with_bounding_boxes(
    source_folders: list[Path],
) -> list[tuple[str, str, dict[str, float], str]]:
    """
    Extract labels with their text content and bounding boxes from source folders.

    For each source folder:
    - Load its results.yaml
    - Identify all labels (keys that are NOT 'collectra_results_metadata')
    - For each label with type 'collectra.Text' or 'collectra.ImageCrop':
        - Extract the label name (the YAML key)
        - Extract the data/text content
        - For Text types: traverse upwards through parents to find the LAST
          parent with type 'collectra.ImageCrop' and get its bounding box
        - For ImageCrop types: use its own bounding box
        - Append tuple: (label_name, text_content, bounding_box_dict, source_folder_path_str)

    Args:
        source_folders: List of paths to collectra folders containing results.yaml files.

    Returns:
        List of tuples containing:
        - label_name: The YAML key (e.g., "registration_number")
        - text_content: The data field value
        - bounding_box: Dict with x_center, y_center, width_relative, height_relative
        - source_folder_path: String path to the source folder

    Example:
        >>> sources = [Path("/data/model1/sample.grapto"), Path("/data/model2/sample.grapto")]
        >>> results = extract_labels_with_bounding_boxes(sources)
        >>> print(results[0])
        ('registration_number', 'P.350015', {'x_center': 0.43, ...}, '/data/model1/sample.grapto')
    """
    all_labels: list[tuple[str, str, dict[str, float], str]] = []

    for source_folder in source_folders:
        try:
            results_data = load_results_yaml(source_folder)
        except FileNotFoundError:
            logger.warning(f"Skipping folder without results.yaml: {source_folder}")
            continue
        except yaml.YAMLError as e:
            logger.warning(f"Skipping folder with invalid YAML: {source_folder}: {e}")
            continue

        source_folder_str = str(source_folder)

        for label_name, entry in results_data.items():
            # Skip metadata
            if label_name == "collectra_results_metadata":
                continue

            # Skip non-dict entries
            if not isinstance(entry, dict):
                continue

            entry_type = entry.get("type", "")
            data_content = entry.get("data", "")

            # Convert data to string if it exists
            if data_content is not None:
                data_content = str(data_content)
            else:
                data_content = ""

            bounding_box: dict[str, float] | None = None

            if entry_type == "collectra.ImageCrop":
                # For ImageCrop, use its own bounding box
                bounding_box = _get_bounding_box_from_entry(entry)

            elif entry_type == "collectra.Text":
                # For Text, traverse up to find last ImageCrop parent
                bounding_box = _find_parent_with_bounding_box(entry, results_data)

            # Only add entries that have valid bounding boxes
            if bounding_box is not None:
                all_labels.append(
                    (label_name, data_content, bounding_box, source_folder_str)
                )
                logger.debug(
                    f"Extracted label '{label_name}' with bounding box from {source_folder_str}"
                )

    logger.info(
        f"Extracted {len(all_labels)} labels with bounding boxes "
        f"from {len(source_folders)} source folders"
    )

    return all_labels


def calculate_iou(box1: dict[str, float], box2: dict[str, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Both boxes are expected in center format with relative coordinates:
    {x_center, y_center, width_relative, height_relative} with values in 0-1 range.

    Args:
        box1: First bounding box dict with center format.
        box2: Second bounding box dict with center format.

    Returns:
        IoU value between 0.0 (no overlap) and 1.0 (identical boxes).

    Example:
        >>> box1 = {"x_center": 0.5, "y_center": 0.5, "width_relative": 0.2, "height_relative": 0.2}
        >>> box2 = {"x_center": 0.5, "y_center": 0.5, "width_relative": 0.2, "height_relative": 0.2}
        >>> calculate_iou(box1, box2)
        1.0
    """
    # Handle edge cases: zero-size boxes
    if box1["width_relative"] <= 0 or box1["height_relative"] <= 0:
        return 0.0
    if box2["width_relative"] <= 0 or box2["height_relative"] <= 0:
        return 0.0

    # Convert center format to corner format (x1, y1, x2, y2)
    box1_x1 = box1["x_center"] - box1["width_relative"] / 2
    box1_y1 = box1["y_center"] - box1["height_relative"] / 2
    box1_x2 = box1["x_center"] + box1["width_relative"] / 2
    box1_y2 = box1["y_center"] + box1["height_relative"] / 2

    box2_x1 = box2["x_center"] - box2["width_relative"] / 2
    box2_y1 = box2["y_center"] - box2["height_relative"] / 2
    box2_x2 = box2["x_center"] + box2["width_relative"] / 2
    box2_y2 = box2["y_center"] + box2["height_relative"] / 2

    # Calculate intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    # Calculate intersection area
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Calculate union area
    box1_area = box1["width_relative"] * box1["height_relative"]
    box2_area = box2["width_relative"] * box2["height_relative"]
    union_area = box1_area + box2_area - inter_area

    # Handle edge case: both boxes have zero area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def calculate_centroid_box(boxes: list[dict[str, float]]) -> dict[str, float]:
    """
    Calculate the centroid bounding box from a list of bounding boxes.

    Computes the average of all 4 fields (x_center, y_center, width_relative,
    height_relative) across all provided boxes.

    Args:
        boxes: List of bounding box dicts in center format.

    Returns:
        Centroid bounding box dict with averaged values.

    Raises:
        ValueError: If boxes list is empty.

    Example:
        >>> boxes = [
        ...     {"x_center": 0.4, "y_center": 0.4, "width_relative": 0.1, "height_relative": 0.1},
        ...     {"x_center": 0.6, "y_center": 0.6, "width_relative": 0.3, "height_relative": 0.3}
        ... ]
        >>> calculate_centroid_box(boxes)
        {'x_center': 0.5, 'y_center': 0.5, 'width_relative': 0.2, 'height_relative': 0.2}
    """
    if not boxes:
        raise ValueError("Cannot calculate centroid from empty list of boxes")

    n = len(boxes)
    return {
        "x_center": sum(b["x_center"] for b in boxes) / n,
        "y_center": sum(b["y_center"] for b in boxes) / n,
        "width_relative": sum(b["width_relative"] for b in boxes) / n,
        "height_relative": sum(b["height_relative"] for b in boxes) / n,
    }


def group_by_bounding_box(
    labels: list[tuple[str, str, dict[str, float], str]],
    iou_threshold: float = 0.6,
) -> list[list[tuple[str, dict[str, float], str]]]:
    """
    Group extracted labels by bounding box similarity using IoU.

    Groups labels based on spatial overlap of their bounding boxes. Labels with
    IoU above the threshold are grouped together. When multiple entries from
    the same source folder would be added to a group, a conflict resolution
    strategy is applied: the entry with higher IoU to the group's centroid is kept.

    Args:
        labels: List of tuples (label_name, text_content, bounding_box, source_folder_path)
            as returned by extract_labels_with_bounding_boxes().
        iou_threshold: Minimum IoU value to consider boxes as belonging to the
            same group (default: 0.6).

    Returns:
        List of groups, where each group is a list of tuples:
        (text_content, bounding_box, source_folder_path).

    Example:
        >>> labels = [
        ...     ("reg", "P.350015", {"x_center": 0.5, ...}, "/path1"),
        ...     ("reg", "P.350015", {"x_center": 0.51, ...}, "/path2"),
        ... ]
        >>> groups = group_by_bounding_box(labels, iou_threshold=0.6)
        >>> len(groups)  # Boxes with high overlap form one group
        1
    """
    if not labels:
        return []

    # Each group is a list of tuples: (text_content, bounding_box, source_folder_path)
    grouping_list: list[list[tuple[str, dict[str, float], str]]] = []

    for label_name, text_content, bounding_box, source_folder_path in labels:
        assigned_to_group = False

        for group in grouping_list:
            # Calculate max IoU between current box and ALL boxes in the group
            max_iou = 0.0
            for _, group_box, _ in group:
                iou = calculate_iou(bounding_box, group_box)
                max_iou = max(max_iou, iou)

            if max_iou > iou_threshold:
                # Check if group already has an entry from the same source_folder_path
                existing_entry = None
                existing_idx = None
                for idx, (existing_text, existing_box, existing_source) in enumerate(
                    group
                ):
                    if existing_source == source_folder_path:
                        existing_entry = (existing_text, existing_box, existing_source)
                        existing_idx = idx
                        break

                if existing_entry is not None:
                    # Conflict: same source folder already in group
                    # Calculate group centroid (excluding the conflicting entry)
                    other_boxes = [
                        box for _, box, src in group if src != source_folder_path
                    ]

                    if other_boxes:
                        centroid = calculate_centroid_box(other_boxes)

                        # Compare IoUs to centroid
                        existing_iou_to_centroid = calculate_iou(
                            existing_entry[1], centroid
                        )
                        new_iou_to_centroid = calculate_iou(bounding_box, centroid)

                        if new_iou_to_centroid > existing_iou_to_centroid:
                            # Replace existing with new
                            logger.warning(
                                f"Conflict in group: replacing entry from '{source_folder_path}' "
                                f"(old IoU to centroid: {existing_iou_to_centroid:.4f}, "
                                f"new IoU to centroid: {new_iou_to_centroid:.4f})"
                            )
                            group[existing_idx] = (
                                text_content,
                                bounding_box,
                                source_folder_path,
                            )
                        else:
                            # Keep existing, discard new
                            logger.warning(
                                f"Conflict in group: keeping existing entry from '{source_folder_path}' "
                                f"(existing IoU to centroid: {existing_iou_to_centroid:.4f}, "
                                f"new IoU to centroid: {new_iou_to_centroid:.4f})"
                            )
                    else:
                        # Only one entry in group from same source, keep existing
                        logger.warning(
                            f"Conflict in group: only one other entry exists, "
                            f"keeping existing entry from '{source_folder_path}'"
                        )
                else:
                    # No conflict: add the entry to the group
                    group.append((text_content, bounding_box, source_folder_path))

                assigned_to_group = True
                break

        if not assigned_to_group:
            # Create new group with this entry as the first element
            grouping_list.append([(text_content, bounding_box, source_folder_path)])

    return grouping_list


def find_centroid_text(texts: list[str]) -> str:
    """
    Find the centroid text from a list of texts using edit distance.

    The centroid text is the one with the minimum total edit distance to all
    other texts in the list. This is useful for finding the "most representative"
    text when ensembling multiple OCR or LLM outputs.

    Args:
        texts: List of text strings to find the centroid from.

    Returns:
        The text with minimum total edit distance to all others.
        Returns empty string if the list is empty.
        Returns the single text if the list has only one entry.

    Example:
        >>> texts = ["hello", "helo", "helllo"]
        >>> find_centroid_text(texts)
        'hello'  # Has lowest total distance to others
    """
    if not texts:
        return ""

    if len(texts) == 1:
        return texts[0]

    # Calculate total edit distance for each text to all others
    min_total_distance = float("inf")
    centroid_text = texts[0]

    for i, text_i in enumerate(texts):
        total_distance = 0
        for j, text_j in enumerate(texts):
            if i != j:
                total_distance += editdistance.eval(text_i, text_j)

        if total_distance < min_total_distance:
            min_total_distance = total_distance
            centroid_text = text_i

    return centroid_text


def generate_ensembled_values(
    groups: list[list[tuple[str, dict[str, float], str]]],
) -> dict[str, Any]:
    """
    Generate ensembled values for each group of similar items.

    For each group, extracts the text content and finds the centroid text
    using edit distance. Single-entry groups (standalones) use their text
    directly without distance calculation.

    Args:
        groups: List of groups, where each group is a list of tuples
            (text_content, bounding_box, source_folder_path) as returned by
            group_by_bounding_box().

    Returns:
        A dictionary containing:
        - "ensembled_values": list of dicts, each with:
            - "text": the ensembled text value
            - "bounding_box": the centroid bounding box for the group
            - "group_size": number of entries in the group
            - "source_texts": list of all source texts in the group
        - "statistics": dict with:
            - "total_groups": total number of groups processed
            - "standalones": number of groups with only one entry
            - "ensembled": number of groups with multiple entries

    Example:
        >>> groups = [
        ...     [("P.350015", {...}, "/path1"), ("P.350015", {...}, "/path2")],
        ...     [("Species name", {...}, "/path1")],
        ... ]
        >>> result = generate_ensembled_values(groups)
        >>> print(result["statistics"])
        {'total_groups': 2, 'standalones': 1, 'ensembled': 1}
    """
    ensembled_values: list[dict[str, Any]] = []
    standalones = 0
    ensembled = 0

    for group in groups:
        if not group:
            continue

        # Extract text contents from the group
        texts = [text_content for text_content, _, _ in group]

        # Extract bounding boxes and calculate centroid
        bounding_boxes = [bbox for _, bbox, _ in group]
        centroid_box = calculate_centroid_box(bounding_boxes)

        # Find the ensembled text value
        if len(texts) == 1:
            ensembled_text = texts[0]
            standalones += 1
        else:
            ensembled_text = find_centroid_text(texts)
            ensembled += 1

        ensembled_values.append(
            {
                "text": ensembled_text,
                "bounding_box": centroid_box,
                "group_size": len(group),
                "source_texts": texts,
            }
        )

    statistics = {
        "total_groups": len(groups),
        "standalones": standalones,
        "ensembled": ensembled,
    }

    logger.info(
        f"Ensemble statistics: {statistics['total_groups']} total groups, "
        f"{statistics['standalones']} standalones, "
        f"{statistics['ensembled']} ensemble decisions"
    )

    return {
        "ensembled_values": ensembled_values,
        "statistics": statistics,
    }


def write_ensembled_results(
    ensemble_folder: Path,
    ensembled_data: dict[str, Any],
    groups: list[list[tuple[str, dict[str, float], str]]],
) -> Path:
    """
    Write the ensembled values to the ensemble collectra file's results.yaml.

    Creates a results.yaml file in the ensemble folder with the ensembled values
    organized by their positions (using index-based labels since we don't have
    the original label names in the groups).

    Args:
        ensemble_folder: Path to the ensemble collectra folder (e.g., .grapto folder).
        ensembled_data: Dictionary containing "ensembled_values" and "statistics"
            as returned by generate_ensembled_values().
        groups: The original groups list for reference.

    Returns:
        Path to the created/updated results.yaml file.

    Raises:
        PermissionError: If the results.yaml cannot be written.

    Example:
        >>> ensembled_data = generate_ensembled_values(groups)
        >>> results_path = write_ensembled_results(
        ...     Path("/data/ensemble/sample.grapto"),
        ...     ensembled_data,
        ...     groups
        ... )
    """
    results_yaml_path = ensemble_folder / "results.yaml"

    # Build the results.yaml structure
    results_data: dict[str, Any] = {
        "collectra_results_metadata": {
            "workflow": "Ensemble",
            "version": "0.1.0",
            "ensemble_statistics": ensembled_data["statistics"],
        }
    }

    # Add each ensembled value as a field
    for idx, value_data in enumerate(ensembled_data["ensembled_values"]):
        field_key = f"ensembled_field_{idx}"
        results_data[field_key] = {
            "type": "collectra.Text",
            "id": f"ensemble-{idx}",
            "data": value_data["text"],
            "ensemble_info": {
                "group_size": value_data["group_size"],
                "source_texts": value_data["source_texts"],
                "bounding_box": value_data["bounding_box"],
            },
        }

    try:
        with open(results_yaml_path, "w") as f:
            yaml.dump(results_data, f, default_flow_style=False, sort_keys=False)
    except PermissionError as e:
        raise PermissionError(f"Cannot write results.yaml: {results_yaml_path}") from e

    logger.info(f"Wrote ensembled results to: {results_yaml_path}")

    return results_yaml_path


def ensemble_groups_for_file(
    ensemble_collectra_folder: Path,
    link_yaml_path: Path,
    iou_threshold: float = 0.6,
) -> dict[str, Any]:
    """
    Run the full ensemble workflow for a single collectra file.

    This function integrates the complete ensemble pipeline:
    1. Get source collectra files from link.yaml
    2. Extract labels with bounding boxes from all sources
    3. Group labels by bounding box similarity
    4. Generate ensembled values for each group
    5. Write results to the ensemble file's results.yaml

    Args:
        ensemble_collectra_folder: Path to the ensemble collectra folder.
        link_yaml_path: Path to the link.yaml file.
        iou_threshold: Minimum IoU for grouping (default: 0.6).

    Returns:
        Dictionary with ensemble results including:
        - "ensembled_values": list of ensembled text values
        - "statistics": ensemble statistics
        - "results_yaml_path": path to the written results.yaml

    Example:
        >>> result = ensemble_groups_for_file(
        ...     Path("/data/ensemble/sample.grapto"),
        ...     Path("/data/ensemble/link.yaml")
        ... )
        >>> print(result["statistics"])
        {'total_groups': 10, 'standalones': 3, 'ensembled': 7}
    """
    # Step 1: Get source files
    source_folders = get_source_collectra_files(
        ensemble_collectra_folder, link_yaml_path
    )

    logger.info(
        f"Processing ensemble for {ensemble_collectra_folder.name} "
        f"with {len(source_folders)} sources"
    )

    # Step 2: Extract labels with bounding boxes
    labels = extract_labels_with_bounding_boxes(source_folders)

    # Step 3: Group by bounding box
    groups = group_by_bounding_box(labels, iou_threshold=iou_threshold)

    # Step 4: Generate ensembled values
    ensembled_data = generate_ensembled_values(groups)

    # Step 5: Write results
    results_yaml_path = write_ensembled_results(
        ensemble_collectra_folder, ensembled_data, groups
    )

    return {
        "ensembled_values": ensembled_data["ensembled_values"],
        "statistics": ensembled_data["statistics"],
        "results_yaml_path": results_yaml_path,
    }
