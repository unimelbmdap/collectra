from dataclasses import dataclass
from pathlib import Path
from PIL import Image as ImagePil

@dataclass(kw_only=True)
class Image:
    image: str # Path to the image file
    width: int
    height: int

    @classmethod
    def build(cls, path: Path):
        img = ImagePil.open(path)
        width, height = img.size
        return cls(image=str(path), width=width, height=height)

@dataclass(kw_only=True)
class ImageCrop(Image):    
    pass    