"""Tests for collectra.tasks.canonicalisers module"""

from collectra import Text
from collectra.tasks.canonicalisers import LLMCanonicaliser

# =============================================================================
# LLMCanonicaliser.__init__
# =============================================================================


def test_init_defaults():
    task = LLMCanonicaliser(name="test", model="dummy", entities="cat,dog")
    assert task.threshold == 0.8
    assert task.count == 5
    assert task.embedding_model == "text-embedding-3-large"


def test_init_splits_comma_separated_entities():
    task = LLMCanonicaliser(name="test", model="dummy", entities="cat,dog,fish")
    assert task.entities == ["cat", "dog", "fish"]


def test_init_entity_file_path(tmp_path):
    entity_file = tmp_path / "entities.txt"
    entity_file.write_text("cat\ndog\nfish\n")
    task = LLMCanonicaliser(name="test", model="dummy", entities=str(entity_file))
    assert task.entities == entity_file


# =============================================================================
# LLMCanonicaliser.run
# =============================================================================


def test_run_returns_text():
    task = LLMCanonicaliser(
        name="species",
        model="dummy",
        template="{species}",
        entities="cat,dog,fish",
        output="result",
    )
    result = task.run(Text("species", data="cat"))
    assert isinstance(result, Text)
    assert result.name == "result"
    assert isinstance(result.data, str) and result.data
    assert "cat" in result.data


def test_run_passes_entities_to_prompt():
    task = LLMCanonicaliser(
        name="species",
        model="dummy",
        template="{entities}",
        entities="cat,dog,fish",
        output="result",
    )
    result = task.run()
    assert isinstance(result, Text)
    assert "cat, dog, fish" in result.data
