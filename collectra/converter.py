import typer, yaml, zipfile, os, json, pytz, shutil
from rich import print
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing_extensions import Annotated
from typing import List

app = typer.Typer()

def create_collectra_metadata(timestamp=None, validation=False):
    return {
        "timestamp": timestamp or datetime.now(pytz.utc).isoformat(),
        "validation": validation,
    }

def find_items_by_type(graph, item_type):
    return [item for item in graph if item.get("@type") == item_type]

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def copy_file(src_path, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, 'wb') as dest_f:
        with open(src_path, 'rb') as src_f:
            dest_f.write(src_f.read())

def create_bounding_box_item(bbox):
    return {        
        "image": "specimen_sheet",
        "x_center": float(bbox.get("x_center")),
        "y_center": float(bbox.get("y_center")),
        "width_relative": float(bbox.get("width_relative")),
        "height_relative": float(bbox.get("height_relative")),
    }

def get_all_files(paths: List[str]) -> List[Path]:
    files = []
    for path in tqdm(paths, desc="Collecting files"):
        path = Path(path)
        if path.is_dir():
            sub_files = [Path(file) for file in path.glob("**/*.*")]
            files.extend(sub_files)
        elif path.is_file():
            files.append(Path(path))
    return files

@app.command()
def convert(
    paths: Annotated[List[str], typer.Argument(help="List of files to convert")],
    file_format: Annotated[str, typer.Option("--file-format", "-f", help="File format to convert to")] = "hespi"
):
    files = get_all_files(paths)       
    specimen = Path("specimen")
    images = Path("tests/images")
    
    ensure_directory(specimen)
    ensure_directory(images)

    for file in tqdm(files, desc="Converting files"):
        with zipfile.ZipFile(file, 'r') as zipf:
            ensure_directory(specimen / file.stem)
            zipf.extractall(specimen / file.stem)

    for specimen_path in tqdm(specimen.glob("*"), desc="Processing specimens"):        
        # Get json file in the specimen directory
        json_file = next(specimen_path.glob("*.json"), None)
        # Get image file in the specimen directory
        image_file = next(specimen_path.glob("*.jpg"), None) or next(specimen_path.glob("*.png"), None)
        if not json_file or not image_file:
            typer.echo(f"Skipping {specimen_path} as it does not contain both a JSON file and an image file.")
            continue
        
        config = {}
        
        with open(json_file, 'r') as f:
            data = json.load(f)     

        graph = data.get("@graph")

        file_items = find_items_by_type(graph, "File")
        for item in file_items:
            for_validation = item.get("for_validation", False)
            config["collectra_results_metadata"] = create_collectra_metadata(
                item.get("timestamp"), for_validation
            )

        config["specimen_sheet"] = {
            "type": "Image",
            "path": str(image_file.name),
        }        
        
        bounding_box_items = find_items_by_type(graph, "BoundingBox")
        bounding_boxes = [{
            "x_center": item.get("x_center"),
            "y_center": item.get("y_center"),
            "width_relative": item.get("width_relative"),
            "height_relative": item.get("height_relative"),
            "class_id": item.get("class_id"),
        } for item in bounding_box_items]
        
        class_mapping_items = find_items_by_type(graph, "ClassMapping")
        classes = class_mapping_items[0].get("classes", []) if class_mapping_items else []                
        
        for bounding_box in bounding_boxes:
            class_id = bounding_box.get("class_id")
            class_name = classes[int(class_id)]
            item = create_bounding_box_item(bounding_box)                        
            if class_name not in config:
                config[class_name] = {"type": "ImageCrop", "items": [item]}
            else:                                                
                config[class_name]["items"].append(item)

        for class_name in classes:
            if class_name not in config:
                continue
            try:
                if len(config[class_name]["items"]) == 1:                
                    config[class_name] = {
                        **config[class_name],
                        **config[class_name]["items"][0]
                    }
                    config[class_name].pop("items", None)                
            except Exception as e:
                print(config[class_name])
                print(f"Error processing class {class_name}: {e}")                
        
        # Save the config to a YAML file        
        file_root = images / f"{image_file.stem}.{file_format}"
        config_file = file_root / "results.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            for key in config:
                f.write(yaml.dump({key: config[key]}, default_flow_style=False, sort_keys=False))
                f.write("\n")
        image_dest = file_root / image_file.name        
        copy_file(image_file, image_dest)        
    
    if specimen.exists():
        shutil.rmtree(specimen, ignore_errors=True)

if __name__ == "__main__":
    app()

