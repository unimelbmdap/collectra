"""BM25 search over short texts, such as entity names and spreadsheet rows."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from collectra.tasks.bm25_index import BM25Index, Document, tokenize

GENERA = [
    "Monograptus",
    "Climacograptus",
    "Didymograptus",
    "Orthograptus",
    "Dicellograptus",
]


def names(results):
    return [document.page_content for document, _ in results]


def test_tokens_are_words_and_their_character_trigrams():
    assert tokenize("Ab cde") == ["ab", "#ab", "ab#", "cde", "#cd", "cde", "de#"]


def test_tokens_ignore_case_and_accents():
    assert tokenize("Müller") == tokenize("muller")
    assert tokenize("ÉTAGE") == tokenize("étage")


def test_tokens_split_on_punctuation():
    assert tokenize("Acacia, (dealbata)") == tokenize("acacia dealbata")


def test_an_exact_name_comes_first():
    index = BM25Index([str(i) for i in range(len(GENERA))], GENERA)
    results = index.search("Didymograptus", k=3)
    assert names(results)[0] == "Didymograptus"
    assert results[0][0] == Document(id="2", page_content="Didymograptus")


def test_a_misspelt_name_still_finds_the_entity():
    index = BM25Index([str(i) for i in range(len(GENERA))], GENERA)
    assert names(index.search("Monograptas", k=1)) == ["Monograptus"]
    assert names(index.search("Climacograptis", k=1)) == ["Climacograptus"]


def test_rare_words_count_for_more_than_common_ones():
    texts = [
        "Acacia dealbata Victoria",
        "Acacia mearnsii Victoria",
        "Banksia serrata Victoria",
    ]
    index = BM25Index(["1", "2", "3"], texts)
    assert names(index.search("mearnsii Victoria", k=1)) == ["Acacia mearnsii Victoria"]


def test_scores_are_positive_and_highest_first():
    index = BM25Index([str(i) for i in range(len(GENERA))], GENERA)
    scores = [score for _, score in index.search("Orthograptus", k=5)]
    assert all(isinstance(score, float) and score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_texts_that_share_nothing_with_the_query_are_left_out():
    index = BM25Index(["1", "2"], ["Acacia", "Eucalypt"])
    assert index.search("xyz") == []
    assert names(index.search("Acacia Qqq", k=5)) == ["Acacia"]


def test_equal_scores_keep_the_texts_order():
    index = BM25Index(["1", "2", "3"], ["Acacia", "Eucalypt", "Acacia"])
    assert [doc.id for doc, _ in index.search("Acacia", k=5)] == ["1", "3"]


def test_k_limits_the_results():
    index = BM25Index([str(i) for i in range(len(GENERA))], GENERA)
    assert len(index.search("graptus", k=2)) == 2
    assert len(index.search("graptus", k=50)) == len(GENERA)
    assert index.search("graptus", k=0) == []


def test_an_empty_index_finds_nothing():
    index = BM25Index([], [])
    assert len(index) == 0
    assert index.search("anything") == []


def test_a_query_with_no_words_finds_nothing():
    index = BM25Index(["1"], ["Acacia"])
    assert index.search(" , ") == []


def test_ids_and_texts_must_match_up():
    with pytest.raises(ValueError, match="same length"):
        BM25Index(["1"], ["a", "b"])


def test_an_index_round_trips_through_disk(tmp_path):
    path = tmp_path / "index"
    assert not BM25Index.exists(path)
    BM25Index(["1", "2"], ["Acacia", "Banksia"]).save(path)
    assert BM25Index.exists(path)
    loaded = BM25Index.load(path)
    assert loaded.ids == ["1", "2"] and loaded.texts == ["Acacia", "Banksia"]
    assert names(loaded.search("Banksia", k=1)) == ["Banksia"]


def test_the_index_needs_no_langchain():
    """The core install has no langchain: its dependencies do not build on new Pythons."""
    script = textwrap.dedent("""
        import importlib.abc
        import sys

        class WithoutLangChain(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0].startswith('langchain'):
                    raise ModuleNotFoundError(fullname, name=fullname)
        sys.meta_path.insert(0, WithoutLangChain())

        from collectra.tasks.bm25_index import BM25Index
        import collectra.tasks.canonicalisers
        import collectra.tasks.irn_resolvers

        assert BM25Index(["1"], ["apple"]).search("apple")[0][0].id == "1"
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_langchain_is_not_a_dependency():
    import tomllib

    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())[
        "project"
    ]
    requirements = project["dependencies"] + [
        requirement
        for extra in project["optional-dependencies"].values()
        for requirement in extra
    ]
    assert not [r for r in requirements if r.startswith("langchain")]
