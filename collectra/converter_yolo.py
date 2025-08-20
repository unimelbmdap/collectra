import yaml, pytz, shutil, os
import typer as tp
from typing_extensions import Annotated
from pathlib import Path
from datetime import datetime
from pprint import pprint
import tqdm

app = tp.Typer()


def convert_files(config: dict):
    files = []
    with open(config.get("train"), "r") as file:
        files.extend(
            [{"path": line.strip(), "split": "train"} for line in file if line.strip()]
        )
    with open(config.get("val"), "r") as file:
        files.extend(
            [{"path": line.strip(), "split": "val"} for line in file if line.strip()]
        )

    names = config.get("names", [])

    label_paths = []
    for file in files:
        file_str = str(file["path"])
        label_path = Path(
            file_str.replace(".jpg", ".txt")
            .replace(".png", ".txt")
            .replace("images", "labels")
        )
        if label_path.exists():
            label_paths.append(label_path)

    if len(label_paths) != len(files):
        raise ValueError("Mismatch between number of image files and label files.")

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
        output_path = (
            Path.cwd()
            / "images"
            / image.name.replace(".jpg", ".hespi").replace(".png", ".hespi")
        )
        os.makedirs(output_path, exist_ok=True)
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
    yolo_config: Annotated[Path, tp.Argument(help="Path to YOLO configuration file")],
):
    yolo_config = Path(yolo_config)
    if not yolo_config.exists():
        raise FileNotFoundError(
            f"YOLO configuration file '{yolo_config}' does not exist."
        )

    with open(yolo_config, "r") as file:
        config = yaml.safe_load(file)
    convert_files(config)


@app.command()
def cluster(
    yolo_config: Annotated[
        Path, tp.Option("--config", help="Path to YOLO configuration file")
    ],
    is_file_for_validation: Annotated[
        bool, tp.Option("--validation", help="Is file for validation")
    ],
    image_folder: Annotated[
        Path, tp.Option("--image-folder", "-i", help="Path to image folder")
    ],
):
    yolo_config = Path(yolo_config)
    if not yolo_config.exists():
        raise FileNotFoundError(
            f"YOLO configuration file '{yolo_config}' does not exist."
        )

    with open(yolo_config, "r") as file:
        config = yaml.safe_load(file)

    class_counts = {name: 0 for name in config.get("names", [])}

    for image in image_folder.glob("*.hespi"):
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
