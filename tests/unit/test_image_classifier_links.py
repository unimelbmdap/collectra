from types import SimpleNamespace

import yaml

from collectra import Collectra, Image, ImageClassifierYOLO, Link
from collectra.types.base import ArtefactNode


def make_image(tmp_path, name="Palynomorph", filename="image.jpg"):
    from PIL import Image as PillowImage

    path = tmp_path / filename
    PillowImage.new("RGB", (12, 12), "white").save(path)
    return Image(name=name, id=f"{name}15", data=path)


def test_classifier_returns_link_to_input_image(tmp_path):
    image = make_image(tmp_path)
    task = ImageClassifierYOLO(
        "classifier",
        model="unused.pt",
        output=["Algae", "Fungi", "Pollen"],
    )
    task._init_model = lambda: None
    task.model = lambda _: [
        SimpleNamespace(
            probs=SimpleNamespace(top1=1),
            names={0: "Algae", 1: "Fungi", 2: "Pollen"},
        )
    ]

    result = task.run(image)
    result.set_parents([image])

    assert type(result) is Link
    assert isinstance(result, Image)
    assert result.name == "Fungi"
    assert result.target is image
    assert result.get_path() == image.get_path()
    assert result.serialize() == {
        "type": "collectra.Link",
        "id": result.id,
        "parents": "Palynomorph15",
    }


def test_classifier_output_nodes_are_typed_as_links(tmp_path):
    make_image(tmp_path)
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text(
        yaml.safe_dump(
            {
                "collectra_pipeline_metadata": {
                    "name": "Palynomorph",
                    "ext": "palynomorph",
                    "version": "1.0",
                },
                "Palynomorph": {
                    "type": "collectra.Image",
                    "data": "image.jpg",
                },
                "palynomorph_classifier": {
                    "type": "collectra.ImageClassifierYOLO",
                    "model": "yolo11x.pt",
                    "input": "Palynomorph",
                    "output": ["Algae", "Fungi", "Pollen"],
                },
            },
            sort_keys=False,
        )
    )
    pipeline = Collectra.from_file(pipeline_file)
    pipeline.connect()

    for name in ("Algae", "Fungi", "Pollen"):
        node = pipeline.node_manager.resolve_node(name)
        assert isinstance(node, ArtefactNode)
        assert node.types == {Link}


def test_link_loaded_from_yaml_can_bind_to_first_parent(tmp_path):
    image = make_image(tmp_path)
    result_dir = tmp_path / "sample.palynomorph"
    result_dir.mkdir()
    (result_dir / "results.yaml").write_text(
        yaml.safe_dump(
            {
                "collectra_results_metadata": {},
                "Fungi": {
                    "type": "collectra.Link",
                    "id": "Fungi1",
                    "parents": image.id,
                },
            }
        )
    )
    node = ArtefactNode("Fungi", types={Link})
    node.process("Fungi", result_dir)
    link = next(iter(node.items.values()))

    assert type(link) is Link
    assert link.target is None
    assert link.bind({image.id: image}) is image
    assert isinstance(link, Image)


def test_classifier_training_uses_link_node_name_as_class(tmp_path):
    image = make_image(tmp_path)
    link = Link(name="Fungi", id="Fungi1", parents=[image.id])
    task = ImageClassifierYOLO("classifier", model="unused.pt")

    training = task.prepare_training_inputs(
        [],
        [image],
        {"sample": [link]},
        {"sample": [image]},
    )

    assert training == [image]
    assert image.name == "Fungi"
