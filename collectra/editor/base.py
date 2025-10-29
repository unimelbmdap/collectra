import yaml, webview
from dataclasses import dataclass, field
from pathlib import Path

from collectra.utils import change_dir, load_class_from_string

__all__ = ["Viewer"]

@dataclass
class Viewer:

    ROOT_PAGE: str = str(Path.cwd() / "collectra" / "editor" / "index.html")
    file: Path | None = field(default=None)    

    def __post_init__(self):        
        self.index = self.ROOT_PAGE

    @property
    def window(self) -> webview.Window | None:
        return webview.active_window()    
    
    @property
    def ftypes(self) -> tuple[str, str]:
        return ('Collectra Files (*.*)', 'All files (*.*)')
    
    def getFile(self):
        items = self.window.create_file_dialog(
            dialog_type=webview.FileDialog.FOLDER, allow_multiple=False, file_types=self.ftypes
        )
        files: list[Path] = [Path(file) for file in items] if items else []
        return self.loadItems(files)
        

    def getFolder(self):
        items = self.window.create_file_dialog(
            dialog_type=webview.FileDialog.FOLDER, allow_multiple=True, file_types=self.ftypes
        )
        files: list[Path] = [Path(file) for file in items] if items else []
        return self.loadItems(files)
    
    def loadItems(self, files: list[Path] = []) -> dict:        
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
                        item["name"] = key
                        item["data"] = item.pop("path") if "path" in item else item["data"]
                        item_data = type_(**item)
                        value[index]["data"] = item_data.__getstate__()
                    if len(value) == 1:
                        data[key] = value[0]
            file_data[file.name] = data        
        return file_data
        
    def edit(self) -> None:
        pass



