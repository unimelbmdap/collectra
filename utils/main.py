from datetime import datetime
from pathlib import Path
from typing_extensions import Annotated
from rich import print
from rich.progress import track
import os, pytz, re, shutil, tempfile, tqdm, yaml, zipfile,traceback
import typer as tp

app = tp.Typer()


def get_files(config: dict) -> list[dict[str, str]]:
    files = []
    parent_dir = Path(config.get("parent_dir", "."))
    train_file = parent_dir / config.get("train", "")    
    if train_file.is_file():
        with open(train_file, "r") as file:
            files.extend(
                [
                    {"path": parent_dir / line.strip(), "split": "train"}
                    for line in file
                    if line.strip()
                ]
        )
    val_file = parent_dir / config.get("val", "")
    if val_file.is_file():
        with open(val_file, "r") as file:
            files.extend(
                [
                    {"path": parent_dir / line.strip(), "split": "val"}
                    for line in file
                    if line.strip()
                ]
            )    
    return files


def get_label_paths(config: dict, files: list[dict[str, str]]) -> list[Path]:
    label_paths = []
    for file in files:
        file_str = re.sub(r"\.(jpg|png)$", ".txt", str(file["path"]))
        file_str = file_str.replace("images", "labels")
        label_path = Path(file_str)
        if label_path.exists():
            label_paths.append(label_path)
        else:
            print(f"Label file not found for image: {file['path']}")
            print(f"Expected label path: {label_path}")
    if len(label_paths) != len(files):
        raise ValueError("Mismatch between number of image files and label files.")
    return label_paths


def convert_files(config: dict) -> None:
    files = get_files(config)
    names = config.get("names", [])    
    output_dir = Path(config.get("output_dir", "output"))
    format = re.sub(r"[^0-9a-zA-Z]+", "", config.get("format", "grapto").lower())
    label_paths = get_label_paths(config, files)
    for index in tqdm.tqdm(range(len(files)), desc="Converting files"):
        image = Path(files[index]["path"])
        results_yaml = {
            "collectra_results_metadata": {
                "timestamp": datetime.now(pytz.utc).isoformat(),
                "validation": files[index]["split"] == "val",
            },
            "specimen_sheet": {"type": "collectra.Image", "data": image.name},
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
                "type": "collectra.ImageCrop",
                "data": image.name,
                "x_center": float(x_center),
                "y_center": float(y_center),
                "width_relative": float(width_relative),
                "height_relative": float(height_relative),
            }
            if class_name not in results_yaml:
                results_yaml[class_name] = [item_dimensions]
            else:
                results_yaml[class_name].append(item_dimensions)
        for key in results_yaml:
            if (
                key not in ["collectra_results_metadata", "specimen_sheet"]
                and len(results_yaml[key]) == 1
            ):
                results_yaml[key] = results_yaml[key][0]                                    

        output_path = (
            output_dir
            / "images"
            / image.name.replace(".jpg", f".{format}").replace(".png", f".{format}")
        )                
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
    yolo_config: Annotated[
        Path, tp.Option("--config", "-c", help="Path to YOLO configuration file")
    ],
    format: Annotated[
        str,
        tp.Option(
            "--format", "-f", help="File format, only 'yolo' is supported currently"
        ),
    ],
    output_dir: Annotated[
        Path, tp.Option("--output", "-o", help="Output directory for converted files")
    ],
):
    """Convert YOLO formatted dataset to Collectra format
    Args:
        yolo_config (Path): Path to YOLO configuration file
    """
    try:
        yolo_config = Path(yolo_config)
        if not yolo_config.exists():
            raise FileNotFoundError(
                f"YOLO configuration file '{yolo_config}' does not exist."
            )
        with open(yolo_config, "r") as file:
            config = yaml.safe_load(file)
            config["format"] = format
            config["output_dir"] = output_dir
            config["parent_dir"] = yolo_config.parent
            convert_files(config)
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")

@app.command()
def convert_raw_img(
    imf: Annotated[str, tp.Option("--image_format", "-i", help="Image format to be searched for")],
    format: Annotated[str, tp.Option("--format", "-f", help="Output file format")],
    folder: Annotated[
        Path, tp.Option("--folder", "-d", help="Path to folder containing images")
    ],
    first_label: Annotated[
        str, tp.Option("--first-label", "-l", help="First label name")
    ],
):
    from PIL import Image
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)    
    for image in track(folder.glob(f"*.{imf}")):            
        results_yaml = {
            "collectra_results_metadata": {
                "timestamp": datetime.now(pytz.utc).isoformat(),
                "validation": False,
            },
            f"{first_label}" : {
                "type": "collectra.Image", 
                "data": image.name
            },            
        }                        
        file_path = folder_path / f"{image.stem}.{format}"  
        file_path.mkdir(parents=True, exist_ok=True)        
        with open(file_path / "results.yaml", "w") as file:
            for result in results_yaml:
                yaml.dump(
                    {result: results_yaml[result]},
                    file,
                    default_flow_style=False,
                    sort_keys=False,
                )
                file.write("\n")
        img = Image.open(image)
        data = list(img.getdata())
        img_no_exif = Image.new(img.mode, img.size)
        img_no_exif.putdata(data)
        img_no_exif.save(folder_path / f"{image.stem}.{format}" / image.name)           

@app.command()
def cluster(
    yolo_config: Annotated[
        Path, tp.Option("--config", "-c", help="Path to YOLO configuration file")
    ],    
    image_folder: Annotated[
        Path, tp.Option("--image-folder", "-i", help="Path to image folder")
    ],
    format: Annotated[str, tp.Option("--format", "-f", help="File format")],
    is_file_for_validation: Annotated[
        bool, tp.Option("--validation", "-v", help="Is file for validation")
    ] = False,
):
    yolo_config = Path(yolo_config)
    if not yolo_config.exists():
        raise FileNotFoundError(
            f"YOLO configuration file '{yolo_config}' does not exist."
        )

    with open(yolo_config, "r") as file:
        config = yaml.safe_load(file)

    class_counts = {name: 0 for name in config.get("names", [])}    
    num_training_files = 0
    num_validation_files = 0    
    for image in image_folder.glob(f"*.{format}"):
        results_yaml_path = ""
        if image.is_dir():
            results_yaml_path = image / "results.yaml"
        elif image.is_file():
            results_yaml_path = image / "results.yaml"
        with open(results_yaml_path, "r") as file:
            results_yaml = yaml.safe_load(file)

        validation = results_yaml["collectra_results_metadata"]["validation"]
        if validation != is_file_for_validation:        
            num_training_files += 1
            continue        
        
        num_validation_files += 1

        for key in results_yaml:
            if key not in ["collectra_results_metadata", "specimen_sheet"]:
                label = results_yaml[key]
                if key not in class_counts:
                    class_counts[key] = 0
                if isinstance(label, list):
                    class_counts[key] += len(label)
                else:
                    class_counts[key] += 1

    for class_name, count in class_counts.items():
        print(f"Total count for class '{class_name}': {count}")


def convert_image_objects(results_yaml: dict) -> dict:        
    for key in results_yaml:        
        data = results_yaml[key]
        if "type" in data:
            if data["type"] in ["ImageCrop", "Image"]:                            
                if "path" in data:
                    data["data"] = data.pop("path")
                elif "image" in data:
                    data["data"] = results_yaml[data.pop("image")]["data"]
                data["type"] = f"collectra.{data['type']}"
                if "items" in data:
                    for item in data["items"]:                        
                        item["data"] = data["data"]
                        item["type"] = data["type"]
                    results_yaml[key] = data["items"]                                                
        #         data["type"] = "collectra.images.base.ImageCrop"
        #     elif data["type"] == "Image" or data["type"] == "collectra.images.Image":
        #         data["type"] = "collectra.images.base.Image"
        #         image_path = data["path"]
        # if "image" in data:
        #     new_data = {
        #         "type": data["type"],
        #         "path": image_path if image_path else data["image"],
        #     }
        #     data.pop("type", None)
        #     data.pop("image", None)
        #     new_data.update(data)
        #     results_yaml[key] = new_data    
    return results_yaml


def modify_file(path):
    with open(path, "r") as file:
        results_yaml = yaml.safe_load(file)

    results_yaml = convert_image_objects(results_yaml)

    with open(path, "w") as file:
        for result in results_yaml:
            yaml.dump(
                {result: results_yaml[result]},
                file,
                default_flow_style=False,
                sort_keys=False,
            )
            file.write("\n")


def modify_zipfile(path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(path=tmpdir)
            results_yaml_path = tmpdir / "results.yaml"
            modify_file(tmpdir)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    zipf.write(os.path.join(root, file), file)


@app.command()
def modify_results_file(
    format: Annotated[str, tp.Option("--format", "-f", help="File format")],
    path: Annotated[
        Path,
        tp.Option("--path", "-p", help="Path to folder containing results.yaml files"),
    ],
):
    if path.is_dir():
        if "results.yaml" in [f.name for f in path.iterdir()]:
            modify_file(path / "results.yaml")
        else:
            for file in path.glob(f"*.{format}"):
                if file.is_dir():
                    modify_file(file / "results.yaml")
                elif file.is_file():
                    modify_zipfile(file)
    if path.is_file() and path.suffix == f".{format}":
        modify_zipfile(path)


@app.command()
def show(
    path: Annotated[str, tp.Option("--path", "-p", help="collectra file path")],
    label: Annotated[
        str, tp.Option("--label", "-l", help="label of the original image")
    ],
    font_path: Annotated[str, tp.Option("--font", "-f", help="Font to be used for text display")],
    font_size: Annotated[int, tp.Option("--font-size", "-fs", help="Font size to be used for text display")] = 48,
    save: Annotated[bool, tp.Option("--save", help="Save the drawn images")] = False,
    output: Annotated[str, tp.Option("--output", help="The folder to save the detected images")] = "detected_images",
):
    """Display the original image with crops and labels from the results.yaml annotation file.

    Args:
        path (Path): Path to the collectra file containing annotations.

    """    
    from PIL import ImageDraw, ImageFont
    from collectra import Image, ImageCrop
    from collectra.utils import change_dir, load_class_from_string
    file_path = Path(path)
    font = ImageFont.truetype(font_path, size=font_size)
    if not file_path.exists() or not file_path.is_dir():
        raise ValueError("File does not exist or is invalid, exiting...")
    with change_dir(file_path):
        with open("results.yaml", "r") as f:
            data = yaml.safe_load(f)        
        value = data.pop(label, None)        
        if value is None or "type" not in value or load_class_from_string(value["type"]) != Image:
            raise ValueError("This file does not contain a base image to draw on! Exiting...")                
        original_image = Image(name=label, parents=[], data=value["data"])
        image_pil = original_image.pil()
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        draw = ImageDraw.Draw(image_pil)        

        for key, values in data.items():
            if not isinstance(values, list):
                values = [values]
            try:
                for value in values:
                    if not isinstance(value, dict) or "type" not in value or load_class_from_string(value["type"]) != ImageCrop:
                        continue
                    image_crop = original_image.make_crop(
                        name=key,
                        x_center = float(value["x_center"]),
                        y_center = float(value["y_center"]),
                        width_relative = float(value["width_relative"]),
                        height_relative = float(value["height_relative"])
                    )
                    coordinates = image_crop.coordinates()
                    draw.rectangle(coordinates, outline="blue", width=8)                
                    left, upper, _, _ = coordinates
                    upper = upper - font_size*1.1 if upper - font_size*1.1 >= 0 else 0                     
                    draw.text((left, upper), key, fill="red", font=font)    
            except Exception as e:
                print(f"Error processing label '{key}': {e}")
                return
    if save:        
        Path(output).mkdir(parents=True, exist_ok=True)
        image_pil.save(Path(output) / f"{file_path.stem}.jpg")
    else:
        image_pil.show()


@app.command()
def legacy_fix(
    folder: Annotated[
        Path,
        tp.Option(
            "--folder", "-d", help="Path to folder containing legacy collectra files"
        ),
    ],
    format: Annotated[
        str, tp.Option("--format", "-f", help="File format of collectra files")
    ],
):    
    for image in track(folder.glob(f"*.{format}")):            
        results_yaml_path = ""
        if not image.is_dir():
            raise Warning(f"Expected directory for image: {image}")
        results_yaml_path = image / "results.yaml"            
        with open(results_yaml_path, "r") as file:
            results_yaml = yaml.safe_load(file)
        results_yaml = convert_image_objects(results_yaml)
        with open(results_yaml_path, "w") as file:
            for result in results_yaml:
                yaml.dump(
                    {result: results_yaml[result]},
                    file,
                    default_flow_style=False,
                    sort_keys=False,
                )
                file.write("\n")


if __name__ == "__main__":
    app()
