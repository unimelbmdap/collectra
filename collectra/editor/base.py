import yaml, json, webbrowser, webview

from jinja2 import Environment, PackageLoader, select_autoescape
from dataclasses import dataclass
from pathlib import Path

from collectra.utils import change_dir, load_class_from_string

__all__ = ["Editor", "Viewer"]


class Viewer:

    window: webview.Window

    def open_file_dialog(self) -> list[Path]:
        file_types = ('Collectra file (*.grapto)', 'All files (*.*)')

        items = self.window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=True, file_types=file_types
        )        
        files: list[Path] = [Path(file) for file in items] if items else []
        return files
    
    def loadItems(self):
        files: list[Path] = self.open_file_dialog()
        file_data: dict = dict()
        for file in files:
            with change_dir(file):
                with open("results.yaml", "r") as f:
                    data = yaml.safe_load(f) or {}
                for key, value in data.items():
                    if not isinstance(value, list):
                        data[key] = [value]
                        value = data[key]                            
                    for index, item in enumerate(value):   
                        if "type" not in item:
                            continue             
                        type_ = load_class_from_string(item.pop("type"))
                        item_data = type_(key, **item)
                        value[index]["data"] = item_data.__getstate__()
                    if len(value) == 1:
                        data[key] = value[0]
            file_data[file.name] = data
        return file_data
        


@dataclass
class Editor:

    file: Path

    def __post_init__(self):
        self.env = Environment(
            loader=PackageLoader("collectra.editor"), autoescape=select_autoescape()
        )

    def edit(self):
        pass

    def view(self):
        template = self.env.get_template("imagebb.html")
        with change_dir(self.file):
            results = Path("results.yaml")
            if not results.exists():
                raise Exception(f"{self.file} is an invalid file. Exiting...")
            with open(results, "r") as f:
                data = yaml.safe_load(f)
            header = data.pop("collectra_results_metadata", "")
            src_img = ""
            bounding_boxes: list = list()
            text_items: list = list()
            for k, v in data.items():
                if not isinstance(v, list):
                    v = [v]
                for item in v:
                    type_ = item.get("type", "")
                    if type_ in [
                        "collectra.Image",
                        "collectra.ImageCrop",
                        "collectra.types.images.Image",
                        "collectra.types.images.ImageCrop",
                    ]:
                        src_img = item.get("data", src_img)
                        src_img = item.get("path", src_img)
                        keys = list(item.keys())
                        bb_params = set(
                            [
                                "x_center",
                                "y_center",
                                "width_relative",
                                "height_relative",
                            ]
                        )
                        if not bb_params.issubset(keys):
                            continue
                        bb = {bb_param: item[bb_param] for bb_param in bb_params}
                        bb["label"] = k
                        bounding_boxes.append(bb)
                    elif type_ in ["collectra.Text", "collectra.types.texts.Text"]:
                        text = item.get("data", "")
                        text_items.append({"name": k, "text": text})
            # --- Prepare Context for Jinja2 Template ---
            context = {
                "src_img": src_img,
                "bounding_boxes": json.dumps(bounding_boxes),
                "text_items": text_items,
            }

            # --- Render Template ---
            rendered_html = template.render(context)
            viewer_file = Path.cwd() / "viewer.html"
            with open(viewer_file, "w", encoding="utf-8") as f:
                f.write(rendered_html)
            webbrowser.open_new_tab(f"file:///{str(viewer_file)}")
