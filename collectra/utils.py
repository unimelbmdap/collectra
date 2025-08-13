from typing import List
from rich import print
from rocrate.rocrate import ROCrate
from pathlib import Path
from tqdm import tqdm

def success_msg(message: str):
    return f"[green]Success[/green]: {message}"

def error_msg(message: str):
    return f"[red]Error[/red]: {message}"

def processing_msg(message: str):
    return f"[dark_orange]Processing[/dark_orange]: {message}"

def get_class_path(obj_or_class):
    """Get the full dotted path: package.module.ClassName"""
    if hasattr(obj_or_class, '__class__'):
        # It's an instance
        cls = obj_or_class.__class__
    else:
        # It's already a class
        cls = obj_or_class
    
    return f"{cls.__module__}.{cls.__name__}"

class BaseROCrate:
    def __init__(self, name: str, version: str, output: Path, crate: ROCrate = None):
        self.name = name
        self.version = version
        self.output = output
        self.crate = crate     


def get_all_files(data: List[str], file_format: str) -> List[str]:
    """
    Get all files from the provided paths with the specified file format.
    :param data: List of potential file/file paths to search.
    :param file_format: File format to filter by (e.g., '.jpg', '.png').
    :return: List of file paths that match the specified format.
    """
    files: List[Path] = []
    for path in tqdm(data, desc="Collecting files"):
        path = Path(path)
        if path.is_dir():
            sub_files = [Path(file) for file in path.glob(f"**/*{file_format}")]            
            files.extend(sub_files)            
        elif path.is_file() and path.suffix.replace(".", "") == file_format:            
            files.append(Path(path))            
    if len(files) == 0:
        raise Exception(f"No files found with format '{file_format}'")        
    return files

def rcollect(path: Path, files: List[Path], file_format: str) -> None:
    if path.is_dir():
        matches = list(path.glob("results.yaml"))
        if matches:
            if len(matches) > 1:
                raise ValueError(f"Found multiple results.yaml files: {matches}")
            files.append(path)
        else:
            for path in path.glob(f"**/*{file_format}"):
                rcollect(path, files=files, file_format=file_format)
    if path.is_file() and path.suffix.replace(".", "") == file_format:
        files.append(path)