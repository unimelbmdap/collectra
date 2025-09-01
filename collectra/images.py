from dataclasses import dataclass, field

@dataclass(kw_only=True)
class Image:
    image: str # Path to the image file
    width: int
    height: int

@dataclass(kw_only=True)
class ImageCrop(Image):    
    pass    