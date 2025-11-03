import yaml, webview
from dataclasses import dataclass, field
from pathlib import Path

from collectra.utils import change_dir, load_class_from_string

__all__ = ["Viewer"]

@dataclass
class Viewer:

    ROOT_PAGE: str = str(Path(__file__).parent / "index.html")
    file: Path | None = field(default=None)    

    def __post_init__(self):        
        self.index = self.ROOT_PAGE

    @property
    def window(self) -> webview.Window | None:
        return webview.active_window()    
    
    @property
    def ext(self) -> tuple[str, str]:
        return ('Collectra Files (*.*)', 'All files (*.*)')
    
    def getFile(self):
        items = self.window.create_file_dialog(
            dialog_type=webview.FileDialog.FOLDER, allow_multiple=False, file_types=self.ext
        )
        files: list[Path] = [Path(file) for file in items] if items else []
        return self.loadItems(files)
        

    def getFolder(self):
        items = self.window.create_file_dialog(
            dialog_type=webview.FileDialog.FOLDER, allow_multiple=True, file_types=self.ext
        )
        files: list[Path] = [Path(file) for file in items] if items else []
        return self.loadItems(files)
    
    def _loadInstance(self, data: dict) -> dict:
        for key, value in data.items():
            if not isinstance(value, list):
                data[key] = [value]
                value = data[key]                            
            for item in value:   
                if "type" not in item:
                    continue             
                item_data = item.copy()
                type_ = load_class_from_string(item_data.pop("type"))
                item_data["name"] = key
                item_data["data"] = item_data.pop("path") if "path" in item_data else item_data["data"]
                item_data = type_(**item_data)
                item["data"] = item_data.__getstate__()
            if len(value) == 1:
                data[key] = value[0]
        return data

    def _loadItem(self, file: Path) -> dict:
        with change_dir(file):
            with open("results.yaml", "r") as f:
                data = yaml.safe_load(f) or dict()
            data = self._loadInstance(data)
        return data

    def loadItems(self, files: list[Path] = []) -> dict:
        file_data: dict = dict()
        for file in files:            
            file_data[file.name] = self._loadItem(file)
        return file_data
        
    def edit(self) -> None:
        pass



