import os
from typing import List
from rich import print
from rocrate.rocrate import ROCrate
from pathlib import Path

class BaseROCrate:
    def __init__(self, name: str, version: str, output: Path, crate: ROCrate = None):
        self.name = name
        self.version = version
        self.output = output
        self.crate = crate     