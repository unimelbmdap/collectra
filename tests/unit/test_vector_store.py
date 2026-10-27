import numpy as np
import pytest

from collectra.tasks.vector_store import VectorStore


class FakeEmbeddings:
    """Embeds a text as a one-hot-ish vector keyed by its first letter."""

    def __init__(self):
        self.calls = 0

    def _vector(self, text):
        vector = np.zeros(4, dtype=np.float32)
        vector["abcd".index(text[0].lower())] = 1.0
        return vector.tolist()

    def embed_documents(self, texts):
        self.calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def store(tmp_path):
    return VectorStore.build(
        tmp_path / "index.embed",
        FakeEmbeddings(),
        ["apple", "avocado", "banana", "cherry"],
        ["1", "2", "3", "4"],
        "fake-model",
    )


def test_search_orders_by_squared_l2_distance(store):
    results = store.similarity_search_with_score("apricot", k=3)

    assert [doc.id for doc, _ in results[:2]] == ["1", "2"]  # exact matches, tie by order
    assert [round(score, 6) for _, score in results] == [0.0, 0.0, 2.0]
    assert results[0][0].page_content == "apple"


def test_k_larger_than_store_returns_everything(store):
    assert len(store.similarity_search_with_score("banana", k=50)) == 4


def test_store_round_trips_through_disk(store, tmp_path):
    path = tmp_path / "index.embed"
    assert VectorStore.exists(path)
    assert VectorStore.exists(path, "fake-model")
    assert not VectorStore.exists(path, "another-model")

    embeddings = FakeEmbeddings()
    loaded = VectorStore.load(path, embeddings)

    assert embeddings.calls == 0  # nothing re-embedded
    assert loaded.similarity_search_with_score("cat", k=1)[0][0].page_content == "cherry"


def test_missing_or_foreign_directory_is_not_a_store(tmp_path):
    assert not VectorStore.exists(tmp_path / "nothing")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "chroma.sqlite3").touch()  # a previous Chroma index
    assert not VectorStore.exists(tmp_path / "old")


def test_empty_store_searches_to_nothing(tmp_path):
    empty = VectorStore.build(tmp_path / "empty", FakeEmbeddings(), [], [])
    assert empty.similarity_search_with_score("anything") == []
    assert VectorStore.exists(tmp_path / "empty")
    assert VectorStore.load(tmp_path / "empty", FakeEmbeddings()).similarity_search_with_score("x") == []


def test_mismatched_texts_and_ids_rejected(tmp_path):
    with pytest.raises(ValueError):
        VectorStore.build(tmp_path / "bad", FakeEmbeddings(), ["a"], ["1", "2"])


def test_canonicaliser_builds_then_reuses_its_index(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    from collectra.tasks.canonicalisers import LLMCanonicaliser

    fake_module = ModuleType("langchain_openai")
    fake_module.OpenAIEmbeddings = lambda model: FakeEmbeddings()
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fruit.txt").write_text("apple\nbanana\ncherry\napple\n")
    task = LLMCanonicaliser("fruit", "fake-model", entities="fruit.txt", count=3)

    first = task.retrieve_entities("apricot")
    assert (tmp_path / "fruit_index.embed" / "vectors.npy").exists()
    assert first["apple"] == ["0", "3"]  # both apple rows, grouped by text

    # Second call loads from disk; the file is no longer needed.
    (tmp_path / "fruit.txt").unlink()
    assert task.retrieve_entities("apricot") == first
