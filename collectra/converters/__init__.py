from pathlib import Path

from .yolo import YOLOConverter

converters = {
    "yolo": YOLOConverter,
}


def convert_files(folder: Path, converter_name: str) -> None:
    converter_cls = converters.get(converter_name)
    if converter_cls is None:
        raise ValueError(
            f"Converter '{converter_name}' not found. Available converters: {list(converters.keys())}"
        )
    converter = converter_cls(folder)
    converter.convert()
