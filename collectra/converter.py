import typer, yaml, zipfile, os, json
from rich import print
from pathlib import Path
from rocrate.model.data_entity import DataEntity
from rocrate.model.contextentity import ContextEntity
from rocrate.model.file import File
from rocrate.rocrate import ROCrate
from tqdm import tqdm
from typing_extensions import Annotated
from typing import List

app = typer.Typer()

class ImageBoundingBox(DataEntity):
    """
    Represents an image bounding box in a grapto file.
    """

    def __init__(self, crate, identifier=None, properties=None):
        super(ImageBoundingBox, self).__init__(crate, identifier, properties)

    def _empty(self):
        val = {
            "@id": self.id,
            "@type": "BoundingBox",
            "x_center": None,
            "y_center": None,
            "width_relative": None,
            "height_relative": None,
            "class_id": None,
        }
        return val

class ClassMapping(DataEntity):
    """
    Represents a label mapping in a grapto file.
    """

    def __init__(self, crate, identifier=None, properties=None):
        super(ClassMapping, self).__init__(crate, identifier, properties)

    def _empty(self):
        val = {
            "@id": self.id,
            "@type": "ClassMapping",
        }
        return val

class CroppingEngine(DataEntity):
    """
    Represents a cropping engine in a grapto file.
    """

    def __init__(self, crate, identifier=None, properties=None):
        super(CroppingEngine, self).__init__(crate, identifier, properties)

    def _empty(self):
        val = {
            "@id": self.id,
            "@type": "CroppingEngine",
        }
        return val

def process_config(config_file: Path) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    with open(config_file) as stream:
        try:
            config = yaml.safe_load(stream)
            training_file = Path(config_file.stem) / config.get("train")
            validation_file = Path(config_file.stem) / config.get("val")
            classes = config.get("names", [])
            with open(training_file, 'r') as train_f, open(validation_file, 'r') as val_f:
                train_files =  [line.strip() for line in train_f.readlines() if line.strip()]
                train_label_files = [file.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt") for file in train_files]
                val_files = [line.strip() for line in val_f.readlines() if line.strip()]
                val_label_files = [file.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt") for file in val_files]
            return train_files, train_label_files, val_files, val_label_files, classes
        except yaml.YAMLError as exc:
            typer.echo(f"Error reading the configuration file: {exc}")
            raise

def convert_format(files: list[str], label_files: list[str], classes: list[str], file_format: str, output: Path, validation: bool = False):    
    for i in tqdm(range(len(files))):        
        file = Path(files[i])                
        if not file.exists():                
            typer.echo(f"File {file} does not exist. Skipping...")
            continue
        new_crate = ROCrate()        
        image = new_crate.add_file(file, properties={
            "@id": f"{file.name}",                        
            "encoding_format": f"{file.suffix}",
            "bounding_boxes": [],          
            "for_validation": validation,                          
            "name": ""
        })
        label_file = Path(label_files[i])     

        if label_file.exists():
            with open(label_file, 'r') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
            bbox_list = []
            id_counter = 0
            for label in labels:
                data = label.split(" ")
                properties = {
                    "x_center": data[1],
                    "y_center": data[2],
                    "width_relative": data[3],
                    "height_relative": data[4],
                    "class_id": data[0],                    
                }
                image_bbox = ImageBoundingBox(
                    new_crate,
                    f"bounding_box_{id_counter}",
                    properties=properties
                )
                bbox_list.append(new_crate.add(image_bbox))   
                id_counter += 1
            if len(bbox_list) == 0:
                typer.echo(f"No bounding boxes found in {label_file}. Skipping...")                
            image["bounding_boxes"] = bbox_list     
            cropping_engine = CroppingEngine(
                new_crate,
                "cropping_engine",
                properties={
                    "method": "manual",
                    "version": None
                }
            )
            new_crate.add(cropping_engine)
        else:
            typer.echo(f"Label file {label_file} does not exist. Skipping bounding boxes for {file}.")

        class_mapping_entity = ClassMapping(
            new_crate,
            "class_mapping",
            properties={
                "classes": classes,
            },
        )        
        new_crate.add(class_mapping_entity)        
        new_crate.write_zip(output / f"{file.stem}.{file_format}")               

def handler(config_file: Path, file_format: str = "grapto", output: Path = Path.cwd()):
    """
    Convert a file to HESPI format.
    
    Args:
        input_file (str): The path to the input file.
        file_format (str): The format to convert the file to, default is "grapto".

    """
    train_files, train_label_files, val_files, val_label_files, classes = process_config(config_file)
    convert_format(train_files, train_label_files, classes, file_format, output)
    convert_format(val_files, val_label_files, classes, file_format, output, validation=True)

@app.command()
def convert(    
    config_file: Path = typer.Argument(None, help="Path to the configuration file."),    
    file_format: str = typer.Option("grapto", help="Format to convert the file"),
    output: Path = typer.Option(Path.cwd(), help="Output directory for the converted files.")
):
    handler(config_file, file_format, output)

@app.command()
def convert(
    paths: Annotated[List[str], typer.Argument(help="List of files to convert")],
):
    files = []
    for path in tqdm(paths, desc="Collecting files"):
        path = Path(path)
        if path.is_dir():
            sub_files = [Path(file) for file in path.glob("**/*.*")]
            files.extend(sub_files)
        elif path.is_file():            
            files.append(Path(path))        

    specimen = Path("specimen")
    images = Path("images")
    os.makedirs(specimen, exist_ok=True)
    os.makedirs(images, exist_ok=True)
    for file in tqdm(files, desc="Converting files"):
        with zipfile.ZipFile(file, 'r') as zipf:
            os.makedirs(specimen / file.stem, exist_ok=True)
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

        for item in graph:
            if item.get("@type") == "File":
                for_validation = item.get("for_validation", False)
                if for_validation:
                    config["collectra_results_metadata"] = {
                        "timestamp": item.get("timestamp", "2023-10-01T12:00:00Z"),
                        "validation": True,
                    }
                else:
                    config["collectra_results_metadata"] = {
                        "timestamp": item.get("timestamp", "2023-10-01T12:00:00Z"),
                        "validation": False,
                    }

        config["specimen_sheet"] = {
            "type": "Image",
            "path": str(image_file.name),
        }        

        graph = data.get("@graph")
        bounding_boxes = []
        classes = []
        for item in graph:
            if item.get("@type") == "BoundingBox":
                bbox = {
                    "x_center": item.get("x_center"),
                    "y_center": item.get("y_center"),
                    "width_relative": item.get("width_relative"),
                    "height_relative": item.get("height_relative"),
                    "class_id": item.get("class_id"),
                }
                bounding_boxes.append(bbox)                
        for item in graph:            
            if item.get("@type") == "ClassMapping":
                classes = item.get("classes", [])
                break                
        
        for bounding_box in bounding_boxes:
            class_id = bounding_box.get("class_id")
            class_name = classes[int(class_id)]
            if class_name not in config:
                config[class_name] = {
                    "items": [
                        {
                            "image": "specimen_sheet",
                            "type": "ImageCrop",
                            "x_center": bounding_box.get("x_center"),
                            "y_center": bounding_box.get("y_center"),
                            "width_relative": bounding_box.get("width_relative"),
                            "height_relative": bounding_box.get("height_relative"),
                        }
                    ]                    
                }
            else:
                config[class_name]["items"].append({
                    "image": "specimen_sheet",
                    "type": "ImageCrop",
                    "x_center": bounding_box.get("x_center"),
                    "y_center": bounding_box.get("y_center"),
                    "width_relative": bounding_box.get("width_relative"),
                    "height_relative": bounding_box.get("height_relative"),
                })
        
        # Save the config to a YAML file
        config_file = images / specimen_path / "results.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)        
        image_dest = images / specimen_path / image_file.name
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        with open(image_dest, 'wb') as img_f:
            with open(image_file, 'rb') as src_f:
                img_f.write(src_f.read())           

if __name__ == "__main__":
    app()

