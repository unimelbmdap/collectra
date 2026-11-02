import pytest
import torch
from PIL import Image as PILImage

from collectra import Image, ImageCrop
from collectra.tasks.machine_learning.orienters import ImageOrienter
from collectra.types.images import Orientation


class FakeEfficientNet(torch.nn.Module):
    """Calls black images north and white images west, and records batch sizes."""

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, x):
        self.batch_sizes.append(x.shape[0])
        logits = torch.zeros(x.shape[0], 4)
        logits[torch.arange(x.shape[0]), (x[:, 0].mean(dim=(1, 2)) > 0).long()] = 1
        return logits


@pytest.fixture(autouse=True)
def fake_model_class(monkeypatch):
    monkeypatch.setattr("torchvision.models.EfficientNet", FakeEfficientNet)


@pytest.fixture
def make_image(tmp_path):
    def make(name, colour):
        path = tmp_path / f"{name}.png"
        PILImage.new("RGB", (40, 30), colour).save(path)
        return Image(name=name, data=path)

    return make


def orienter(**kwargs):
    return ImageOrienter(
        "orient", model=FakeEfficientNet(), output="oriented", **kwargs
    )


def test_run_orients_one_image(make_image):
    task = orienter()
    result = task.run(make_image("a", "white"))
    assert result.orientation == Orientation.WEST
    assert result.name == "oriented"
    assert task.model.batch_sizes == [1]
    assert task.run(make_image("b", "black")).orientation == Orientation.NORTH


def test_run_keeps_the_crop_of_a_crop(tmp_path):
    PILImage.new("RGB", (40, 30), "white").save(tmp_path / "c.png")
    crop = ImageCrop(
        name="c",
        data=tmp_path / "c.png",
        x_center=0.5,
        y_center=0.5,
        width_relative=0.5,
        height_relative=0.25,
    )
    result = orienter().run(crop)
    assert isinstance(result, ImageCrop)
    assert (result.x_center, result.width_relative) == (0.5, 0.5)
    assert result.orientation == Orientation.WEST


def test_run_takes_exactly_one_image(make_image):
    with pytest.raises(ValueError, match="single Image"):
        orienter().run()
    with pytest.raises(ValueError, match="single Image"):
        orienter().run(make_image("a", "white"), make_image("b", "black"))


def test_prefetch_predicts_in_batches_of_batch_size(make_image):
    task = orienter(batch_size=2)
    images = [
        make_image("a", "white"),
        make_image("b", "black"),
        make_image("c", "white"),
    ]
    task.prefetch([[image] for image in images])
    assert task.model.batch_sizes == [2, 1]


def test_run_uses_the_prefetched_orientation_without_the_model(make_image):
    task = orienter(batch_size=2)
    white, black = make_image("a", "white"), make_image("b", "black")
    task.prefetch([[white], [black]])
    assert task.run(black).orientation == Orientation.NORTH
    assert task.run(white).orientation == Orientation.WEST
    assert task.model.batch_sizes == [2]


def test_a_prefetched_orientation_is_used_once(make_image):
    task = orienter(batch_size=2)
    white = make_image("a", "white")
    task.prefetch([[white]])
    task.run(white)
    task.run(white)
    assert task.model.batch_sizes == [1, 1]


def test_prefetch_skips_entries_that_are_not_one_image(make_image):
    task = orienter(batch_size=4)
    white = make_image("a", "white")
    task.prefetch([[], [white, white], [white]])
    assert task.model.batch_sizes == [1]


def test_clear_prefetch_discards_the_cached_orientations(make_image):
    task = orienter(batch_size=2)
    white = make_image("a", "white")
    task.prefetch([[white]])
    task.clear_prefetch()
    task.run(white)
    assert task.model.batch_sizes == [1, 1]


def test_prefetch_tells_crops_of_one_image_apart(tmp_path):
    PILImage.new("RGB", (40, 30), "white").save(tmp_path / "c.png")

    def crop(x):
        return ImageCrop(
            name="c",
            data=tmp_path / "c.png",
            x_center=x,
            y_center=0.5,
            width_relative=0.2,
            height_relative=0.2,
        )

    task = orienter(batch_size=2)
    task.prefetch([[crop(0.3)], [crop(0.6)]])
    assert task.model.batch_sizes == [2]
    task.run(crop(0.3))
    task.run(crop(0.6))
    assert task.model.batch_sizes == [2]


def test_the_model_is_loaded_from_a_weights_file_once(monkeypatch, make_image):
    loaded = []

    class Weights(FakeEfficientNet):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(), torch.nn.Linear(8, 1000)
            )

        def load_state_dict(self, state):
            loaded.append(state)

    monkeypatch.setattr(
        "torchvision.models.efficientnet_v2_s", lambda weights: Weights()
    )
    monkeypatch.setattr(torch, "load", lambda path, map_location: {"weights": path})
    task = ImageOrienter("orient", model="weights.pt", output="oriented")
    assert task.run(make_image("a", "white")).orientation == Orientation.WEST
    assert task.run(make_image("b", "black")).orientation == Orientation.NORTH
    assert loaded == [{"weights": "weights.pt"}]
    assert task.model.classifier[1].out_features == 4


def test_images_with_transparency_are_placed_on_white(tmp_path):
    path = tmp_path / "t.png"
    PILImage.new("RGBA", (40, 30), (0, 0, 0, 0)).save(path)
    task = orienter()
    assert task.run(Image(name="t", data=path)).orientation == Orientation.WEST
