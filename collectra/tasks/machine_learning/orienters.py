from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import PrivateAttr

from collectra.types.images import Image, ImageCrop, Orientation

from ..base import Task
from ..object_detection.batching import image_cache_key

if TYPE_CHECKING:
    from torchvision.models import EfficientNet

__all__ = ["ImageOrienter"]


class ImageOrienter(Task):

    model: Any  # a path or a loaded torchvision EfficientNet
    _prefetched: dict = PrivateAttr(default_factory=dict)

    def __init__(self, name, model: str | Path | EfficientNet, **kwargs):
        super().__init__(name, model=model, **kwargs)
        self.transforms = None

    def _init_model(self):
        """Initialisation code here is emulated from https://github.com/duartebarbosadev/deep-image-orientation-detection"""
        import torch
        import torchvision.transforms as transforms
        from torchvision.models import EfficientNet

        image_size = 384
        self.transforms = transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        if isinstance(self.model, EfficientNet):
            self.model.to(device)
            self.device = device
            return

        import torchvision.models as models

        model = models.efficientnet_v2_s(weights=None)

        num_ftrs = model.classifier[1].in_features

        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(num_ftrs, 4),
        )

        state_dict = torch.load(self.model, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self.model = model
        self.device = device

    def _processed_image(self, img: Image):
        from PIL import Image as PILImage

        image = img.pil()

        if image.mode in ("RGB", "L"):
            return image.convert("RGB")

        converted_image = image.convert("RGBA")
        background = PILImage.new("RGBA", converted_image.size, (255, 255, 255))
        background.paste(converted_image, mask=converted_image)
        return background.convert("RGB")

    def _predict_batch(self, images: list[Image]) -> list[Orientation]:
        """The orientation of each image, from one pass of the model over all of them."""
        import torch

        batch = torch.stack(
            [self.transforms(self._processed_image(image)) for image in images]
        ).to(self.device)
        with torch.no_grad():
            predicted = torch.max(self.model(batch), 1)[1]
        return [Orientation(int(index)) for index in predicted]

    @staticmethod
    def _prefetch_key(image: Image) -> tuple:
        return image_cache_key(image)

    def prefetch(self, entries: list[list]) -> None:
        """Orient many images in batches of ``batch_size``, caching the results for ``run``."""
        self._init_model()
        images = [
            entry[0]
            for entry in entries
            if len(entry) == 1 and isinstance(entry[0], Image)
        ]
        self._prefetched = {}
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            for image, orientation in zip(batch, self._predict_batch(batch)):
                self._prefetched[self._prefetch_key(image)] = orientation

    def clear_prefetch(self) -> None:
        self._prefetched = {}

    def run(self, *args: Image) -> Image:
        if len(args) != 1:
            raise ValueError("This task only supports a single Image input.")
        self._init_model()
        img = args[0]
        orientation = self._prefetched.pop(self._prefetch_key(img), None)
        if orientation is None:
            orientation = self._predict_batch([img])[0]

        name = self.get_output_name()

        if isinstance(img, ImageCrop):
            new_image = ImageCrop(
                name=name,
                data=img.data,
                x_center=float(img.x_center),
                y_center=float(img.y_center),
                width_relative=float(img.width_relative),
                height_relative=float(img.height_relative),
                orientation=orientation,
            )
            return new_image

        new_image = Image(
            name=name,
            data=img.data,
            orientation=orientation,
        )

        return new_image
