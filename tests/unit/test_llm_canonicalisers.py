"""The canonicaliser and the IRN resolver ask their model through the LLM task."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from collectra.tasks.canonicalisers import LLMCanonicaliser
from collectra.tasks.irn_resolvers import IRNResolver
from collectra.types.texts import Text
from tests.unit.scripted_llm import Scripted, prompt_parts


def test_the_canonicaliser_puts_its_entities_into_the_prompt():
    model = Scripted(["Acacia"])
    task = LLMCanonicaliser(
        "genus_canonical",
        "m",
        entities="Acacia,Banksia",
        template="Read: {genus}\nChoose from: {entities}",
    )
    task.__dict__["llm"] = model
    result = task.run(Text(name="genus", data="Acaca"))
    assert result() == "Acacia"
    assert prompt_parts(model.requests[0]) == [
        "Read:",
        "Acaca",
        "Choose from:",
        "Acacia, Banksia",
    ]


class FakeStore:
    def __init__(self, *rows):
        self.rows = rows

    def similarity_search_with_score(self, query, k):
        return [
            (SimpleNamespace(id=irn, page_content=card), score)
            for irn, card, score in self.rows
        ]


@pytest.fixture
def resolver(monkeypatch, tmp_path):
    embeddings = ModuleType("langchain_openai")
    embeddings.OpenAIEmbeddings = lambda model: None
    monkeypatch.setitem(sys.modules, "langchain_openai", embeddings)
    monkeypatch.setattr(IRNResolver, "_build_indices", lambda self: None)

    def make(model, *rows):
        task = IRNResolver(
            "irn",
            "m",
            source=tmp_path / "source.xlsx",
            card_template="{name}",
            query_template="{genus}",
            template="Query: {genus}\nCandidates:\n{entities}",
        )
        task.__dict__["llm"] = model
        task._vector_stores = [FakeStore(*rows)]
        return task

    return make


def test_the_irn_resolver_asks_the_model_to_pick_a_candidate(resolver):
    model = Scripted(['"12"'])
    task = resolver(
        model, ("12", "Acacia dealbata", 0.1), ("40", "Acacia mearnsii", 0.3)
    )
    assert task.run(Text(name="genus", data="Acacia dealbata"))() == "12"
    parts = prompt_parts(model.requests[0])
    assert "IRN 12: Acacia dealbata\nIRN 40: Acacia mearnsii" in parts


def test_the_irn_resolver_ignores_an_answer_that_is_not_a_candidate(resolver):
    model = Scripted(["99"])
    task = resolver(model, ("12", "Acacia dealbata", 0.1))
    assert task.run(Text(name="genus", data="Acacia"))() == ""
