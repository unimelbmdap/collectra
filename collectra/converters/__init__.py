from pathlib import Path

from .yolo import YOLOClassifierCSV, YOLOConverter

converters = {
    "yolo": YOLOConverter,
    "yoloclassifier_csv": YOLOClassifierCSV,
}


def convert_files(
    input: Path,
    root_label: str,
    ext: str,
    converter_name: str,
    output: Path,
    force: bool = False,
) -> None:
    """Convert files using the specified converter.

    Args:
        input (Path): Path to the configuration file or folder for conversion.
        root_label (str): Root label for converted data.
        ext (str): File extension for converted collectra files.
        converter_name (str): Name of the converter to use.
        output (Path): Output folder for converted files.
        force (bool, optional): Whether to force overwrite existing files. Defaults to False.

    Raises:
        ValueError: If the specified converter is not found.
    """
    converter_cls = converters.get(converter_name)
    if converter_cls is None:
        raise ValueError(
            f"Converter '{converter_name}' not found. Available converters: {list(converters.keys())}"
        )

    converter = converter_cls(input, root_label, ext, output, force=force)
    converter.convert()
