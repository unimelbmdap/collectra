"""Images without objects are kept as background samples in detection training."""

import yaml
from PIL import Image as PillowImage

from collectra import Image, ImageCrop, ObjectDetectionRFDETR, ObjectDetectionYOLO
from collectra.tasks.machine_learning.training import print_distribution_table
from collectra.types.base import ArtefactNode


def result_dir(tmp_path, annotations):
    path = tmp_path / "item.palynomorph"
    path.mkdir()
    image_path = tmp_path / "image.jpg"
    PillowImage.new("RGB", (12, 12), "white").save(image_path)
    data = {
        "collectra_results_metadata": {"partition": "training"},
        "image": {"type": "collectra.Image", "id": "image", "data": str(image_path)},
        **annotations,
    }
    (path / "results.yaml").write_text(yaml.safe_dump(data))
    return path


def test_empty_annotation_list_is_a_background_image(tmp_path):
    path = result_dir(tmp_path, {"Palynomorph": []})
    nodes = [ArtefactNode("Palynomorph", types={ImageCrop})]

    assert ArtefactNode.batch_process(path, nodes) == []
    samples = ArtefactNode.batch_process(path, nodes, include_unlabelled=True)
    assert [type(sample) for sample in samples] == [Image]


def test_yolo_background_image_gets_empty_label_and_is_trained(tmp_path):
    image_path = tmp_path / "background.jpg"
    PillowImage.new("RGB", (12, 12), "white").save(image_path)
    image = Image(name="image", data=image_path, partition="training")
    log = tmp_path / "log"
    log.mkdir()

    train, val = ObjectDetectionYOLO("detector")._prepare_assets(
        ["Palynomorph"], log, "validation", "", image
    )

    assert train == ["background.jpg"]
    assert val == []
    assert (log / "background.txt").read_text() == ""


def test_rfdetr_background_image_is_kept(tmp_path):
    image_path = tmp_path / "background.jpg"
    PillowImage.new("RGB", (12, 12), "white").save(image_path)
    image = Image(name="image", data=image_path, partition="training")

    train, val = ObjectDetectionRFDETR("detector")._prepare_assets(
        ["Palynomorph"], "validation", "", image
    )

    assert [sample["boxes"] for sample in train] == [[]]
    assert val == []


def test_distribution_table_summarises_background_images(capsys):
    print_distribution_table(
        "Class Distribution",
        ["Palynomorph"],
        {"Palynomorph": 3},
        {"Palynomorph": 1},
        train_images=(2, 5),
        validation_images=(1, 1),
    )
    output = capsys.readouterr().out
    assert "Background images" in output
    assert "Total images" in output
    assert "7" in output
