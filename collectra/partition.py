from pathlib import Path

from .commons.files import CollectraFile


def get_files(args: list[str]) -> tuple[list[Path], list[str]]:
    """Gets the list of files to be partitioned from the command-line arguments. It also returns a reduced list of arguments that excludes the file paths, which can be used for further processing (e.g., extracting partition specifications).

    Args:
        args (list[str]): The list of command-line arguments.

    Raises:
        ValueError: If an unexpected option is encountered.
        ValueError: If input files have mismatched extensions.
    Returns:
        tuple[list[Path], list[str]]: A tuple containing the list of file paths and the reduced list of arguments.
    """

    reduced_args: list[str] = []

    files: list[Path] = []
    idx_skip = -1
    for idx, arg in enumerate(args):
        if idx == idx_skip:
            reduced_args.append(arg)
            continue
        if arg.startswith("--"):
            idx_skip = idx + 1
            reduced_args.append(arg)
            continue
        if arg.startswith("-"):
            raise ValueError(f"Unexpected option {arg}")
        path = Path(arg)
        if path.exists():
            files.append(path)

    current_suffix = ""
    for file in files:
        if current_suffix == "":
            current_suffix = file.suffix
        elif file.suffix != current_suffix:
            raise ValueError(
                f"All input files must have the same extension. Found mismatch at {file.name}"
            )

    return files, reduced_args


def get_partitions(args: list[str]) -> dict[str, str | int | float]:
    """Extracts partition specifications from the command-line arguments.

    Args:
        args (list[str]): The list of command-line arguments.

    Raises:
        ValueError: If a partition value is missing or invalid.

    Returns:
        dict[str, str | int | float]: A dictionary mapping partition labels to their values.
    """
    partitions: dict[str, str | int | float] = {}
    for idx, arg in enumerate(args):
        if arg.startswith("--"):
            label = arg.replace("--", "")
            idx += 1  # Move to the next argument which should be the value for this partition
            value = args[idx] if idx < len(args) else None
            if not value or value.startswith("-"):
                raise ValueError(
                    f"Expected a value for partition {label}, but got {value}"
                )
            partitions[label] = value
    return partitions


def get_partition_value_type(value: str | int | float) -> str:
    """Determines the type of the partition value (percentage, integer, or float) based on its format.

    Args:
        value (str | int | float): The partition value to be evaluated.

    Raises:
        ValueError: If the value does not match any expected type.
    """
    if isinstance(value, str) and value.endswith("%"):
        return "percentage"
    try:
        int(value)  # Check if it can be converted to an integer
        return "integer"
    except ValueError:
        try:
            float(value)  # Check if it can be converted to a float
            return "float"
        except ValueError:
            raise ValueError(
                f"Invalid value for partition: {value}. Must be an integer, float, or percentage."
            )


def validate_partition_value(
    value: str | int | float, expected_type: str
) -> int | float:
    """Validates and converts the partition value to the appropriate type based on the expected type.

    Args:
        value (str | int | float): The partition value to be validated.
        expected_type (str): The expected type of the partition value ("percentage", "integer", or "float").

    Raises:
        ValueError: If the value does not match the expected type or format.

    Returns:
        int | float: The validated and converted partition value.
    """
    match expected_type:
        case "percentage":
            if not (isinstance(value, str) and value.endswith("%")):
                raise ValueError(f"Expected a percentage value, but got {value}")
            return float(
                int(value.replace("%", "")) / 100
            )  # Remove the percentage sign for further processing
        case "integer":
            return int(
                value
            )  # Convert to integer if it's a string representation of an integer
        case "float":
            return float(
                value
            )  # Convert to float if it's a string representation of a float
        case _:
            raise ValueError(f"Unknown expected type: {expected_type}")


def validate_partition_sum(
    validated_partitions: dict[str, int | float],
    value_type: str | None,
    files: list[Path],
):
    """Validates that the sum of partition values is appropriate based on the value type (percentage, integer, or float).

    Args:
        validated_partitions (dict[str, int | float]): The dictionary of validated partition values.
        value_type (str | None): The type of the partition values ("percentage", "integer", or "float").
        files (list[Path]): The list of files to be partitioned

    Raises:
        ValueError: If the total percentage exceeds 100%.
        ValueError: If the total count of partitions exceeds the number of files.
        ValueError: If the total fraction of partitions exceeds 1.
        ValueError: If the value type is unknown.

    """
    values: list[int | float] = list(validated_partitions.values())
    match value_type:
        case "percentage":
            total_percentage = sum(values)
            if total_percentage > 1.0:
                raise ValueError(
                    f"Total percentage of partitions must not exceed 100%. Currently, it sums to {total_percentage * 100}%."
                )
        case "integer":
            total_count = sum(values)
            if total_count > len(files):
                raise ValueError(
                    f"Total count of partitions ({total_count}) cannot exceed the number of files ({len(files)})."
                )
        case "float":
            total_fraction = sum(values)
            if total_fraction > 1.0:
                raise ValueError(
                    f"Total fraction of partitions cannot exceed 1. Currently, it sums to {total_fraction}."
                )
        case _:
            raise ValueError(f"Unknown value type: {value_type}")


def assign_partition_label(file: Path, label: str):
    """Assigns the specified partition label to the given file by updating its metadata.

    Args:
        file (Path): The file to which the partition label will be assigned.
        label (str): The partition label to be assigned to the file.

    """
    collectra_file = CollectraFile.from_data(file)
    collectra_file.collectra_results_metadata.partition = label
    collectra_file.save()


def assign_partition_label_to_files(files: list[Path], label: str):
    """Assigns the specified partition label to the given list of files by updating their metadata.

    Args:
        files (list[Path]): The list of files to which the partition label will be assigned.
        label (str): The partition label to be assigned to the files.

    Raises:
        Exception: If there is an error in assigning the partition label to any of the files, reverts any changes made to the files before the error occurred to maintain consistency.

    """
    old_labels = list()
    try:
        for file in files:
            old_labels.append({"file": file, "label": label})
            assign_partition_label(file, label)
    except Exception as e:
        # Revert any changes made to files in case of an error
        for item in old_labels:
            file = item["file"]
            label = item["label"]
            assign_partition_label(file, label)
        raise e


def partition_files(
    files: list[Path],
    partitions: dict[str, int | float],
    value_type: str | None,
    seed: int,
):
    """Partitions the list of files based on the specified partition values and their type. It shuffles the files using the provided random seed to ensure reproducibility.

    Args:
        files (list[Path]): The list of files to be partitioned.
        partitions (dict[str, int | float]): The dictionary of partition labels and their corresponding values.
        value_type (str | None): The type of the partition values ("percentage", "integer", or "float").
        seed (int): The random seed for shuffling the files.

    Raises:
        ValueError: If the value type is not determined before partitioning.
        ValueError: If an unknown value type is encountered during partitioning.

    """
    import random

    if value_type is None:
        raise ValueError("Value type must be determined before partitioning files.")

    random.seed(seed)
    random.shuffle(files)
    total_files = len(files)

    for label, value in partitions.items():
        match value_type:
            case "percentage":
                num_files = int(value * total_files)
            case "integer":
                num_files = int(value)
            case "float":
                num_files = int(value * total_files)
            case _:
                raise ValueError(f"Unknown value type: {value_type}")
        partition_files = files[:num_files]
        files = files[num_files:]
        print(f"Partition '{label}': {len(partition_files)} files")
        assign_partition_label_to_files(partition_files, label)


def process_partitions(
    files: list[Path], partitions: dict[str, str | int | float], seed: int
):
    """Processes the partition specifications by validating their values, ensuring they sum up appropriately, and then partitioning the files accordingly.

    Args:
        files (list[Path]): The list of files to be partitioned.
        partitions (dict[str, str | int | float]): The dictionary of partition labels and their corresponding values (as strings, integers, or floats).
        seed (int): The random seed for shuffling the files.

    Raises:
        ValueError: If there is an error in processing any of the partition specifications.

    """
    value_type: str | None = None
    validated_partitions: dict[str, int | float] = {}
    for label, value in partitions.items():
        try:
            if value_type is None:
                value_type = get_partition_value_type(value)
            validated_partitions[label] = validate_partition_value(value, value_type)
        except ValueError as e:
            raise ValueError(f"Error processing partition {label}: {e}")
    validate_partition_sum(validated_partitions, value_type, files)
    partition_files(files, validated_partitions, value_type, seed)
