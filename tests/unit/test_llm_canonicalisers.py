"""The canonicaliser and the IRN resolver find candidates with BM25, then ask the LLM."""

import pandas as pd
import pytest

from collectra.tasks.bm25_index import BM25Index
from collectra.tasks.canonicalisers import LLMCanonicaliser
from collectra.tasks.irn_resolvers import (
    NO_MATCH_INSTRUCTION,
    IRNResolver,
    normalise_column_name,
    render_card,
)
from collectra.types.texts import Text
from tests.unit.scripted_llm import Scripted, prompt_parts


def prompt(model, request=0) -> str:
    """The text of a request's prompt, its parts on separate lines."""
    return "\n".join(
        p for p in prompt_parts(model.requests[request]) if isinstance(p, str)
    )


# --- LLMCanonicaliser --------------------------------------------------------------


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


@pytest.fixture
def genera(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "genera.txt"
    path.write_text(
        "Monograptus\nClimacograptus\n\nDidymograptus\nOrthograptus\nMonograptus\n"
    )
    return path


def canonicaliser(entities, model, **kwargs):
    task = LLMCanonicaliser(
        "genus_canonicaliser",
        "m",
        entities=str(entities),
        template="Entities:\n{entities}\n\nDraft text:\n{genus_draft}",
        **kwargs,
    )
    task.__dict__["llm"] = model
    return task


def test_the_canonicaliser_offers_the_closest_entities_from_its_file(genera):
    model = Scripted(["Monograptus"])
    task = canonicaliser(genera, model, count=3)
    result = task.run(Text(name="genus_draft", data="Monograptas"))
    assert result() == "Monograptus"
    entities = prompt_parts(model.requests[0])[1].split(", ")
    assert entities[0] == "Monograptus"
    assert len(entities) == 2


def test_an_entity_listed_twice_is_offered_once_with_both_lines(genera):
    task = canonicaliser(genera, Scripted([]))
    found = task.retrieve_entities("Monograptus")
    assert found["Monograptus"] == ["0", "4"]


def test_the_entity_file_is_indexed_once_and_nothing_is_written(genera, tmp_path):
    task = canonicaliser(genera, Scripted([]))
    task.retrieve_entities("Monograptus")
    index = task._index
    genera.write_text("Changed\n")
    task.retrieve_entities("Orthograptus")
    assert task._index is index
    assert sorted(p.name for p in tmp_path.iterdir()) == ["genera.txt"]


def test_a_draft_that_matches_nothing_offers_no_entities(genera):
    model = Scripted(["''"])
    task = canonicaliser(genera, model)
    assert task.run(Text(name="genus_draft", data="zzz"))() == ""
    assert "graptus" not in prompt(model)


def test_the_old_embedding_model_setting_is_accepted_and_ignored(genera):
    task = canonicaliser(genera, Scripted([]), embedding_model="text-embedding-3-large")
    assert "Monograptus" in task.retrieve_entities("Monograptus")


# --- IRNResolver: building its indexes ------------------------------------------------

ROWS = pd.DataFrame(
    {
        "IRN": [12, 40, None, 12, 77, 90],
        "Genus Name (current)": [
            "Acacia",
            "Acacia",
            "Banksia",
            "Acacia",
            "Banksia",
            None,
        ],
        "Species:": ["dealbata", "mearnsii", "serrata", "dealbata", "serrata", None],
    }
)


@pytest.fixture
def workbook(tmp_path, monkeypatch):
    """A source file, read as ROWS; ``reads`` counts how often it is read."""
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"version 1")
    reads = []

    def read_excel(path):
        reads.append(path)
        return ROWS.copy()

    monkeypatch.setattr(pd, "read_excel", read_excel)
    return source, reads


def resolver(source, model=None, **kwargs):
    kwargs.setdefault("template", "Query: {genus}\nCandidates:\n{entities}")
    task = IRNResolver(
        "irn",
        "m",
        source=source,
        card_template="{genus_name} {species}",
        query_template="{genus}",
        **kwargs,
    )
    task.__dict__["llm"] = model or Scripted([])
    return task


def test_each_row_with_an_irn_and_a_card_is_indexed_once(workbook):
    source, _ = workbook
    task = resolver(source)
    (index,) = task._indexes
    assert index.ids == ["12", "40", "77"]
    assert index.texts == ["Acacia dealbata", "Acacia mearnsii", "Banksia serrata"]
    assert list(task.df.columns) == ["irn", "genus_name", "species"]


def test_filters_use_the_spreadsheets_own_column_names(workbook):
    source, _ = workbook
    task = resolver(source, index_filters=["`Genus Name (current)` == 'Banksia'", None])
    assert [index.ids for index in task._indexes] == [["77"], ["12", "40", "77"]]


def test_the_irn_column_may_be_given_by_its_normalised_name(workbook):
    source, _ = workbook
    assert resolver(source, irn_column="irn")._indexes[0].ids == ["12", "40", "77"]


def test_a_missing_irn_column_is_reported(workbook):
    source, _ = workbook
    with pytest.raises(ValueError, match="'Catalogue' .* not found"):
        resolver(source, irn_column="Catalogue")


def test_indexes_are_reused_until_the_source_changes(workbook):
    source, reads = workbook
    resolver(source)
    task = resolver(source)
    assert len(reads) == 1
    assert task._indexes[0].ids == ["12", "40", "77"]
    source.write_bytes(b"version 2")
    resolver(source)
    assert len(reads) == 2


@pytest.mark.parametrize("missing", ["source", "card_template", "query_template"])
def test_the_resolver_needs_its_source_and_templates(tmp_path, missing):
    kwargs = dict(source=tmp_path / "s.xlsx", card_template="c", query_template="q")
    del kwargs[missing]
    with pytest.raises(ValueError, match=missing):
        IRNResolver("irn", "m", **kwargs)


def test_score_floor_is_refused_with_its_replacement_named(workbook):
    source, _ = workbook
    with pytest.raises(ValueError, match="min_score"):
        resolver(source, score_floor=0.5)


def test_the_no_match_instruction_follows_any_preamble(workbook):
    source, _ = workbook
    assert resolver(source).preamble == NO_MATCH_INSTRUCTION
    assert resolver(source, preamble="Be careful.").preamble == (
        f"Be careful.\n\n{NO_MATCH_INSTRUCTION}"
    )


# --- IRNResolver: resolving ------------------------------------------------------------


def test_the_irn_resolver_asks_the_model_to_pick_a_candidate(workbook):
    source, _ = workbook
    model = Scripted(['"12"'])
    task = resolver(source, model)
    assert task.run(Text(name="genus", data="Acacia dealbata"))() == "12"
    text = prompt(model)
    order = [text.index(f"IRN {irn}:") for irn in ("12", "40", "77")]
    assert order == sorted(order)  # Banksia shares only the trigram "ia#"


def test_the_irn_resolver_ignores_an_answer_that_is_not_a_candidate(workbook):
    source, _ = workbook
    task = resolver(source, Scripted(["99"]))
    assert task.run(Text(name="genus", data="Acacia"))() == ""


@pytest.mark.parametrize("answer", ['""', "none", "N/A", "  "])
def test_an_empty_answer_means_no_match(workbook, answer):
    source, _ = workbook
    task = resolver(source, Scripted([answer]))
    assert task.run(Text(name="genus", data="Acacia"))() == ""


def test_a_query_that_matches_nothing_does_not_ask_the_model(workbook):
    source, _ = workbook
    model = Scripted([])
    assert resolver(source, model).run(Text(name="genus", data="zzz qqq"))() == ""
    assert model.requests == []


def test_a_best_score_below_min_score_does_not_ask_the_model(workbook):
    source, _ = workbook
    model = Scripted([])
    task = resolver(source, model, min_score=1000.0)
    assert task.run(Text(name="genus", data="Acacia dealbata"))() == ""
    assert model.requests == []


def test_too_close_a_runner_up_does_not_ask_the_model(workbook):
    source, _ = workbook
    model = Scripted(["12"])
    task = resolver(source, model, score_gap_eps=1000.0)
    assert task.run(Text(name="genus", data="Acacia"))() == ""
    assert model.requests == []
    task = resolver(source, model, score_gap_eps=0.01)
    assert task.run(Text(name="genus", data="Acacia dealbata"))() == "12"


def test_filters_are_tried_in_order_until_one_matches(workbook):
    source, _ = workbook
    model = Scripted(["none", "40"])
    task = resolver(
        source, model, index_filters=["`Genus Name (current)` == 'Banksia'", None]
    )
    assert task.run(Text(name="genus", data="Banksia Acacia mearnsii"))() == "40"
    assert len(model.requests) == 2
    assert "IRN 77: Banksia serrata" in prompt(model, 0)
    assert "IRN 40: Acacia mearnsii" in prompt(model, 1)


def test_retrieve_entities_lists_candidate_irns_and_cards(workbook):
    source, _ = workbook
    task = resolver(source, count=2)
    assert task.retrieve_entities("Acacia dealbata") == [
        ("12", "Acacia dealbata"),
        ("40", "Acacia mearnsii"),
    ]


# --- IRNResolver: helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Genus Name (current)", "genus_name"),
        ("Species:", "species"),
        ("Locality / Site", "locality_site"),
        ("  Date-Collected  ", "datecollected"),
        ("__A__B__", "a_b"),
    ],
)
def test_column_names_are_normalised(raw, expected):
    assert normalise_column_name(raw) == expected


def test_cards_leave_out_missing_values_and_spare_separators():
    template = "{genus} | {species}, {author} | {year}"
    row = {"genus": "Acacia", "species": float("nan"), "author": None, "year": 1822}
    assert render_card(template, row) == "Acacia | 1822"
    assert render_card("| {a} |", {"a": "x"}) == "x"
    assert render_card("{a}{missing}", {"a": "x"}) == "x"
    assert render_card("{a}", {"a": pd.NA}) == ""
    assert render_card("{a}", {"a": [1, 2]}) == "[1, 2]"
