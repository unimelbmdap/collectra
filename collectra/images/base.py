from collectra.utils import crop
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image as ImagePil
import base64, io

@dataclass(kw_only=True)
class Image:

    img_path: str # Path to the image file
    width: int
    height: int
    format: str | None

    @staticmethod
    def build(path: Path):   
        with ImagePil.open(path) as imf:
            format = imf.format
            width, height = imf.size           
        return Image(img_path=str(path), width=width, height=height, format=format)
    
    @staticmethod
    def is_image_file(path: Path) -> bool:
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
        return path.suffix.lower() in image_extensions
    
    def path(self) -> Path:
        return Path(self.img_path)
    
    def load(self) -> bytes:
        with open(self.img_path, "rb") as img_file:
            buffer = img_file.read()
        return buffer

    def get_encoding(self) -> str:        
        buffer = self.load()
        return base64.b64encode(buffer).decode('utf-8')

    def mime(self) -> str:
        return f"image/{self.format.lower()}" if self.format else "image"      

    def metadata(self) -> dict:
        return {
            "type": f"{self.__class__.__module__}.{self.__class__.__name__}",  
            "path": Path(self.img_path).name,                      
        }


@dataclass(kw_only=True)
class ImageCrop(Image):   

    x_center: float
    y_center: float
    width_relative: float
    height_relative: float

    @staticmethod
    def build(path: Path, **kwargs):        
        img = Image.build(path)
        
        return ImageCrop(
            img_path=str(path), 
            width=img.width, 
            height=img.height, 
            format=img.format, 
            **kwargs
        )

    def coordinates(self) -> tuple[float, float, float, float]:        
        x_center = self.x_center * self.width
        y_center = self.y_center * self.height
        actual_width_half = self.width * self.width_relative / 2
        actual_height_half = self.height * self.height_relative / 2
        left = max(0, min(int(x_center - actual_width_half), self.width))
        upper = max(0, min(int(y_center - actual_height_half), self.height))
        right = max(0, min(int(x_center + actual_width_half), self.width))
        bottom = max(0, min(int(y_center + actual_height_half), self.height))
        return (left, upper, right, bottom)

    def load(self) -> bytes:                
        img = crop(path=self.path(), coordinates=self.coordinates())
        buffer = io.BytesIO()
        img.save(buffer, format=self.format)
        buffer.seek(0)        
        return buffer.read()
    
    def metadata(self) -> dict:
        data = super().metadata()
        data.update({
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width_relative": self.width_relative,
            "height_relative": self.height_relative,
        })
        return data
