from pathlib import Path

from torchvision.models import EfficientNet

from collectra.types.images import Image, ImageCrop, Orientation

from .base import MachineLearningTask

__all__ = ["ImageOrienter"]


class ImageOrienter(MachineLearningTask):

    model: str | Path | EfficientNet

    def __init__(self, name, model: str | Path | EfficientNet, **kwargs):
        super().__init__(name, model, **kwargs)
        self._init_model()
        import torchvision.transforms as transforms

        IMAGE_SIZE = 384
        self.transforms = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def train(self, **kwargs):
        raise NotImplementedError("Training not implemented for ImageOrienter.")

    def _init_model(self):
        """Initialisation code here is emulated from https://github.com/duartebarbosadev/deep-image-orientation-detection"""
        import torch

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
        return background

    def run(self, *args: Image) -> Image:

        import torch

        if len(args) != 1:
            raise ValueError("This task only supports a single Image input.")
        img = args[0]
        image = self._processed_image(img)
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)

        if not isinstance(self.model, EfficientNet):
            raise ValueError("Model must be an instance of EfficientNet.")

        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_class = predicted_idx.item()
            orientation = Orientation(predicted_class)

        name = (
            f"{self.get_name()}_output"
            if not hasattr(self, "output")
            else self.output[0] if isinstance(self.output, list) else self.output
        )

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
