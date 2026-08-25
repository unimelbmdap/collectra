import collectra
from collectra import Artefact, Image, Text
from collectra.types.base import Artefact as BaseArtefact


def test_artefact_is_canonical_base_type():
    assert Artefact is BaseArtefact
    assert issubclass(Image, Artefact)
    assert issubclass(Text, Artefact)


def test_data_is_not_part_of_the_public_api():
    assert "Data" not in collectra.__all__
    assert not hasattr(collectra, "Data")
