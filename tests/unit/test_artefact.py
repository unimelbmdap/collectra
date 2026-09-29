import collectra
from collectra import Artefact, ArtefactNode, Image, Text
from collectra.types.base import Artefact as BaseArtefact
from collectra.types.base import ArtefactNode as BaseArtefactNode


def test_artefact_is_canonical_base_type():
    assert Artefact is BaseArtefact
    assert issubclass(Image, Artefact)
    assert issubclass(Text, Artefact)


def test_data_is_not_part_of_the_public_api():
    assert "Data" not in collectra.__all__
    assert not hasattr(collectra, "Data")


def test_artefact_node_is_canonical_node_type():
    assert ArtefactNode is BaseArtefactNode
    assert "DataNode" not in collectra.__all__
    assert not hasattr(collectra, "DataNode")
