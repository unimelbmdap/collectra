from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing_extensions import Annotated

import pytz, re, shutil, tqdm, yaml
import typer as tp

app = tp.Typer()

def get_files(config: dict) -> list[dict[str, str]]:
    files = []
    parent_dir = Path(config.get("parent_dir", "."))
    train_file = parent_dir / config.get("train", "")
    with open(train_file, "r") as file:
        files.extend(
            [{"path": line.strip(), "split": "train"} for line in file if line.strip()]
        )
    val_file = parent_dir / config.get("val", "")
    with open(val_file, "r") as file:
        files.extend(
            [{"path": line.strip(), "split": "val"} for line in file if line.strip()]
        )
    return files

def get_label_paths(files: list[dict[str, str]]) -> list[Path]:
    label_paths = []
    for file in files:
        file_str = re.sub(r'\.(jpg|png)$', '.txt', str(file["path"]))
        file_str = file_str.replace("images", "labels")
        label_path = Path(file_str)
        if label_path.exists():
            label_paths.append(label_path)    
    if len(label_paths) != len(files):
        raise ValueError("Mismatch between number of image files and label files.")
    return label_paths

def convert_files(config: dict) -> None:
    files = get_files(config)
    names = config.get("names", [])
    output_dir = Path(config.get("output_dir", "output"))
    file_format = re.sub(r'[^0-9a-zA-Z]+', '', config.get("file_format", "grapto").lower())
    label_paths = get_label_paths(files)
    for index in tqdm.tqdm(range(len(files)), desc="Converting files"):
        image = Path(files[index]["path"])
        results_yaml = {
            "collectra_results_metadata": {
                "timestamp": datetime.now(pytz.utc).isoformat(),
                "validation": files[index]["split"] == "val",
            },
            "specimen_sheet": {"type": "Image", "path": image.name},
        }
        label_path = label_paths[index]
        bbox = []
        with open(label_path, "r") as f:
            bboxes = [line.strip() for line in f if line.strip()]
        for bbox in bboxes:
            class_id, x_center, y_center, width_relative, height_relative = bbox.split(
                " "
            )
            class_name = names[int(class_id)]
            item_dimensions = {
                "x_center": float(x_center),
                "y_center": float(y_center),
                "width_relative": float(width_relative),
                "height_relative": float(height_relative),
            }
            if class_name not in results_yaml:
                results_yaml[class_name] = {
                    "type": "ImageCrop",
                    "image": "specimen_sheet",
                    "items": [item_dimensions],
                }
            else:
                results_yaml[class_name]["items"].append(item_dimensions)
        for key in results_yaml:
            if (
                key not in ["collectra_results_metadata", "specimen_sheet"]
                and len(results_yaml[key]["items"]) == 1
            ):
                results_yaml[key] = {
                    **results_yaml[key],
                    **results_yaml[key]["items"][0],
                }
                results_yaml[key].pop("items", None)

        output_path = output_dir / "images" / image.name.replace(".jpg", f".{file_format}").replace(".png", f".{file_format}")      
        output_path.mkdir(parents=True, exist_ok=True)          
        shutil.copyfile(image, output_path / image.name)
        with open(output_path / "results.yaml", "w") as f:
            for key in results_yaml:
                f.write(
                    yaml.dump(
                        {key: results_yaml[key]},
                        default_flow_style=False,
                        sort_keys=False,
                    )
                )
                f.write("\n")


@app.command()
def convert(
    yolo_config: Annotated[Path, tp.Option("--config", "-c", help="Path to YOLO configuration file")],
    file_format: Annotated[str, tp.Option("--format", "-f", help="File format, only 'yolo' is supported currently")],
    output_dir: Annotated[Path, tp.Option("--output", "-o", help="Output directory for converted files")],    
):
    """Convert YOLO formatted dataset to Collectra format
    Args:
        yolo_config (Path): Path to YOLO configuration file
    """
    yolo_config = Path(yolo_config)
    if not yolo_config.exists():
        raise FileNotFoundError(
            f"YOLO configuration file '{yolo_config}' does not exist."
        )

    with open(yolo_config, "r") as file:
        config = yaml.safe_load(file)
        config["file_format"] = file_format
        config["output_dir"] = output_dir
        config["parent_dir"] = yolo_config.parent
        convert_files(config)


@app.command()
def cluster(
    yolo_config: Annotated[
        Path, tp.Option("--config", "-c", help="Path to YOLO configuration file")
    ],
    is_file_for_validation: Annotated[
        bool, tp.Option("--validation", "-v", help="Is file for validation")
    ],
    image_folder: Annotated[
        Path, tp.Option("--image-folder", "-i", help="Path to image folder")
    ],
    file_format: Annotated[str, tp.Option("--format", "-f", help="File format")]
):
    yolo_config = Path(yolo_config)
    if not yolo_config.exists():
        raise FileNotFoundError(
            f"YOLO configuration file '{yolo_config}' does not exist."
        )

    with open(yolo_config, "r") as file:
        config = yaml.safe_load(file)

    class_counts = {name: 0 for name in config.get("names", [])}

    for image in image_folder.glob(f"*.{file_format}"):
        results_yaml_path = ""
        if image.is_dir():
            results_yaml_path = image / "results.yaml"
        elif image.is_file():
            results_yaml_path = image / "results.yaml"
        with open(results_yaml_path, "r") as file:
            results_yaml = yaml.safe_load(file)

        if (
            results_yaml["collectra_results_metadata"]["validation"]
            != is_file_for_validation
        ):
            continue

        for key in results_yaml:
            if key not in ["collectra_results_metadata", "specimen_sheet"]:
                label = results_yaml[key]
                if "items" in label:
                    class_counts[key] += len(label["items"])
                else:
                    class_counts[key] += 1

    for class_name, count in class_counts.items():
        print(f"Total count for class '{class_name}': {count}")


if __name__ == "__main__":
    app()
