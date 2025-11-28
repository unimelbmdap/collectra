from collectra.tasks.base import Task
from collectra.types.images import Image, ImageCrop


class Clusterer:
    def __init__(self, name: str, model: str, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_model = model

    def run(self, *args: Image) -> list[Image]:
        pass