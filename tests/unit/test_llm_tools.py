"""An LLM can call named lookups as tools before it gives its answer."""

from unittest.mock import patch

import pytest
import yaml
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart

from collectra import Pipeline
from collectra.commons import TaskContext
from collectra.tasks.base import Task
from collectra.tasks.llms import LLM, ToolCallingUnsupported
from collectra.types.images import Image
from collectra.types.texts import Text
from tests.unit.scripted_llm import (
    USAGE,
    Scripted,
    answer,
    lookup,
    prompt_text,
    submit,
)
from tests.unit.test_pipeline_stages import files  # noqa: F401

THIS = "tests.unit.test_llm_tools"


def returned(request: list) -> list[ToolReturnPart]:
    """The tool results sent back to the model in its latest request."""
    return [p for p in request[-1].parts if isinstance(p, ToolReturnPart)]


def build(tmp_path, replies=(), tools=("genera_lookup",), **llm):
    (tmp_path / "genera.txt").write_text("Didymograptus\nMonograptus\nClimacograptus\n")
    pipeline = Pipeline(
        "Tools",
        "tls",
        "0.1.0",
        path=str(tmp_path),
        genera_lookup={
            "type": "collectra.EntityLookup",
            "file": "genera.txt",
            "description": "Graptolite genera",
        },
        reader={
            "type": "collectra.LLM",
            "model": "m",
            "template": "{note}",
            "input": "note",
            "output": "answer",
            "tools": list(tools),
            **llm,
        },
    )
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("reader").get_task()
    model = Scripted(list(replies))
    task.__dict__["llm"] = model
    return pipeline, task, model


def note(text="Didymograpus sp."):
    return Text(name="note", data=text)


# --- the option ---------------------------------------------------------------------


def test_tools_accept_a_name_or_a_list_and_default_to_none():
    assert LLM("t", "m", template="x", tools="a").tools == ["a"]
    assert LLM("t", "m", template="x", tools=["a", "b"]).tools == ["a", "b"]
    assert LLM("t", "m", template="x").tools == []
    assert LLM("t", "m", template="x", tools=None).tools == []


def test_the_number_of_lookups_is_limited_and_validated():
    assert LLM("t", "m", template="x").max_tool_calls == 5
    assert LLM("t", "m", template="x", max_tool_calls=2).max_tool_calls == 2
    with pytest.raises(ValueError, match="max_tool_calls"):
        LLM("t", "m", template="x", max_tool_calls=0)


def test_tools_can_be_combined_with_several_outputs():
    task = LLM("t", "m", template="x", output=["a", "b"], tools=["genera_lookup"])
    assert task.tools == ["genera_lookup"] and task.is_structured


# --- the loop -----------------------------------------------------------------------


def test_the_model_looks_things_up_then_answers(tmp_path):
    _, task, model = build(tmp_path, [lookup("Didymograpus"), answer("Didymograptus")])
    result = task.run(note())
    assert isinstance(result, Text) and result() == "Didymograptus"
    assert result.name == "answer"
    assert model.function_tools[0] == ["genera_lookup"]
    assert len(model.requests) == 2


def test_the_lookup_result_is_fed_back_to_the_model(tmp_path):
    _, task, model = build(
        tmp_path, [lookup("Didymograpus", id="abc"), answer("Didymograptus")]
    )
    task.run(note())
    (result,) = returned(model.requests[1])
    assert result.tool_call_id == "abc" and result.tool_name == "genera_lookup"
    assert "Candidates for 'Didymograpus'" in result.content
    assert "Didymograptus" in result.content


def test_a_model_that_needs_no_lookup_just_answers(tmp_path):
    _, task, model = build(tmp_path, [answer("Monograptus")])
    assert task.run(note("Monograptus"))() == "Monograptus"
    assert len(model.requests) == 1


def test_several_lookups_in_one_reply_are_all_answered(tmp_path):
    both = ModelResponse(
        parts=[
            ToolCallPart("genera_lookup", {"query": "didymograpus"}, tool_call_id="a"),
            ToolCallPart("genera_lookup", {"query": "monograptus"}, tool_call_id="b"),
        ],
        usage=USAGE,
    )
    _, task, model = build(tmp_path, [both, answer("done")])
    task.run(note())
    replies = returned(model.requests[1])
    assert [m.tool_call_id for m in replies] == ["a", "b"]
    assert "didymograpus" in replies[0].content and "monograptus" in replies[1].content


def test_the_tasks_own_messages_are_not_polluted_by_the_conversation(tmp_path):
    _, task, _ = build(tmp_path, [lookup("x"), answer("y")])
    task.run(note())
    assert len(task.messages) == 2


def test_an_unknown_tool_name_gets_an_error_the_model_can_read(tmp_path):
    _, task, model = build(tmp_path, [lookup("x", name="made_up"), answer("fine")])
    assert task.run(note())() == "fine"
    message = str(model.requests[1][-1].parts[-1].content)
    assert "made_up" in message and "genera_lookup" in message


def test_a_lookup_that_fails_is_reported_to_the_model_not_raised(tmp_path):
    pipeline, task, model = build(tmp_path, [lookup("x"), answer("fine")])
    with patch.object(
        type(pipeline.resources["genera_lookup"]),
        "search",
        side_effect=RuntimeError("index broke"),
    ):
        assert task.run(note())() == "fine"
    (message,) = returned(model.requests[1])
    assert "Lookup failed" in message.content and "index broke" in message.content


def test_lookups_beyond_the_limit_are_refused_and_the_model_must_answer(tmp_path):
    pipeline, task, model = build(
        tmp_path,
        [lookup("a", "1"), lookup("b", "2"), lookup("c", "3"), answer("final")],
        max_tool_calls=2,
    )
    searches = []
    original = type(pipeline.resources["genera_lookup"]).search

    def spy(self, query, limit=None):
        searches.append(query)
        return original(self, query, limit)

    with patch.object(type(pipeline.resources["genera_lookup"]), "search", spy):
        assert task.run(note())() == "final"
    assert searches == ["a", "b"]  # the third was not run
    (refusal,) = returned(model.requests[3])
    assert (
        "limit" in refusal.content.lower() and "final answer" in refusal.content.lower()
    )


def test_a_model_that_never_stops_calling_tools_is_an_error(tmp_path):
    _, task, _ = build(
        tmp_path, [lookup("x", str(i)) for i in range(20)], max_tool_calls=2
    )
    with pytest.raises(RuntimeError, match="final answer after 2 lookups"):
        task.run(note())


def test_lookups_are_logged_with_the_task_tool_and_query(tmp_path, caplog):
    _, task, _ = build(tmp_path, [lookup("Didymograpus"), answer("x")])
    with caplog.at_level("INFO", logger="collectra"):
        task.run(note())
    line = next(
        r.getMessage() for r in caplog.records if "genera_lookup" in r.getMessage()
    )
    assert "reader" in line and "Didymograpus" in line


# --- configuration problems --------------------------------------------------------


def test_a_model_without_tool_calling_raises_a_clear_error(tmp_path):
    _, task, _ = build(tmp_path, [])
    task.__dict__["llm"] = Scripted([], supports_tools=False)
    with pytest.raises(ToolCallingUnsupported, match=r"reader.*tool"):
        task.run(note())


def test_a_task_not_in_a_pipeline_says_so(tmp_path):
    task = LLM("loose", "m", template="{note}", input="note", tools=["genera_lookup"])
    task.__dict__["llm"] = Scripted([])
    with pytest.raises(RuntimeError, match="loose.*pipeline"):
        task.run(note())


# --- usage ---------------------------------------------------------------------------


def test_usage_is_recorded_for_every_turn_of_the_conversation(tmp_path):
    _, task, _ = build(tmp_path, [lookup("x"), answer("y")])
    task.context = TaskContext(usage_file=tmp_path / "usage.yaml")
    task.run(note())
    usage = yaml.safe_load((tmp_path / "usage.yaml").read_text())["reader"]
    assert (usage["input_tokens"], usage["output_tokens"]) == (6, 4)
    assert usage["total_tokens"] == 10


# --- batching ----------------------------------------------------------------------


def rule(fail_on=None):
    """Look the prompt's text up once, then answer; fail first on ``fail_on``."""

    def respond(messages):
        text = prompt_text(messages)
        looked_up = any(
            isinstance(p, ToolReturnPart)
            for m in messages
            for p in getattr(m, "parts", [])
        )
        if text == fail_on and not looked_up:
            return RuntimeError("rate limited")
        return answer(f"answer:{text}") if looked_up else lookup(text)

    return respond


def RuleModel(fail_on=None, delay=0.0):
    return Scripted(respond=rule(fail_on), delay=delay)


def entries(*texts):
    return [[Text(name="note", data=t)] for t in texts]


def batched(tmp_path, model, size=3):
    _, task, _ = build(tmp_path, [], batch_size=size)
    task.__dict__["llm"] = model
    return task


def test_prefetch_runs_the_conversations_concurrently(tmp_path):
    model = RuleModel(delay=0.02)
    task = batched(tmp_path, model)
    task.prefetch(entries("a", "b", "c"))
    assert len(model.requests) == 6  # two turns each
    assert model.most_at_once == 3
    assert sorted(len(v) for v in task._prefetched.values()) == [1, 1, 1]


def test_run_reuses_the_conversations_without_asking_the_model_again(tmp_path):
    model = RuleModel()
    task = batched(tmp_path, model)
    task.prefetch(entries("a", "b", "c"))
    turns = len(model.requests)
    assert [task.run(*e)() for e in entries("c", "a", "b")] == [
        "answer:c",
        "answer:a",
        "answer:b",
    ]
    assert len(model.requests) == turns


def test_usage_for_prefetched_conversations_is_recorded_when_they_are_used(tmp_path):
    model = RuleModel()
    task = batched(tmp_path, model)
    task.prefetch(entries("a"))
    usage_file = tmp_path / "usage.yaml"
    assert not usage_file.exists()
    task.context = TaskContext(usage_file=usage_file)
    task.run(*entries("a")[0])
    assert yaml.safe_load(usage_file.read_text())["reader"]["input_tokens"] == 6


def test_a_conversation_that_fails_in_prefetch_is_retried_alone(tmp_path):
    model = RuleModel(fail_on="b")
    task = batched(tmp_path, model)
    task.prefetch(entries("a", "b"))
    assert len(task._prefetched) == 1  # only "a" was kept
    assert task.run(*entries("a")[0])() == "answer:a"
    with pytest.raises(RuntimeError, match="rate limited"):  # fails again alone
        task.run(*entries("b")[0])
    assert len(model.requests) >= 3


def test_prefetch_without_tool_support_surfaces_the_error(tmp_path):
    _, task, _ = build(tmp_path, [], batch_size=2)
    task.__dict__["llm"] = Scripted([], supports_tools=False)
    with pytest.raises(ToolCallingUnsupported):
        task.prefetch(entries("a", "b"))


# --- the whole pipeline ----------------------------------------------------------------


class Describe(Task):
    def run(self, image: Image) -> Text:
        return Text(name=self.get_output_name(), data="Didymograpus sp.")


def test_a_pipeline_task_uses_the_lookup_and_saves_the_final_answer(tmp_path, files):
    (tmp_path / "genera.txt").write_text("Didymograptus\nMonograptus\n")
    model = Scripted([lookup("Didymograpus"), answer("Didymograptus")])
    pipeline = Pipeline(
        "Tools",
        "tls",
        "0.1.0",
        path=str(tmp_path),
        genera_lookup={
            "type": "collectra.EntityLookup",
            "file": "genera.txt",
            "description": "Genera",
        },
        first={"type": f"{THIS}.Describe", "input": ["page"], "output": ["note"]},
        reader={
            "type": "collectra.LLM",
            "model": "m",
            "template": "{note}",
            "input": ["note"],
            "output": "answer",
            "tools": ["genera_lookup"],
        },
    )
    with patch.object(LLM, "llm", property(lambda self: model)):
        pipeline.run(files=files[:1])
    saved = yaml.safe_load((tmp_path / "input" / "a.tls" / "results.yaml").read_text())
    assert saved["answer"]["data"] == "Didymograptus"
    assert saved["answer"]["id"] == "answer"
    assert "genera_lookup" not in saved
    assert model.function_tools[0] == ["genera_lookup"]


# --- tools with several outputs ---------------------------------------------------------


def build_structured(tmp_path, replies, **llm):
    (tmp_path / "genera.txt").write_text("Didymograptus\nMonograptus\n")
    pipeline = Pipeline(
        "Tools",
        "tls",
        "0.1.0",
        path=str(tmp_path),
        genera_lookup={
            "type": "collectra.EntityLookup",
            "file": "genera.txt",
            "description": "Graptolite genera",
        },
        reader={
            "type": "collectra.LLM",
            "model": "m",
            "template": "{note}",
            "input": "note",
            "output": ["genus", "age"],
            "tools": ["genera_lookup"],
            **llm,
        },
    )
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("reader").get_task()
    model = Scripted(list(replies))
    task.__dict__["llm"] = model
    return task, model


def test_the_answer_schema_is_offered_as_a_tool_next_to_the_lookups(tmp_path):
    task, model = build_structured(tmp_path, [submit(genus="G", age="A")])
    task.run(note())
    assert model.function_tools[0] == ["genera_lookup"]
    assert model.output_tools[0] == ["LLMOutputs"]


def test_the_model_looks_things_up_then_submits_every_field(tmp_path):
    task, model = build_structured(
        tmp_path,
        [lookup("Didymograpus"), submit(genus="Didymograptus", age="Ordovician")],
    )
    result = task.run(note())
    assert [(t.name, t()) for t in result] == [
        ("genus", "Didymograptus"),
        ("age", "Ordovician"),
    ]
    assert len(model.requests) == 2
    (fed_back,) = returned(model.requests[1])
    assert "Didymograptus" in fed_back.content


def test_a_field_left_out_is_an_empty_text(tmp_path):
    task, _ = build_structured(tmp_path, [submit(genus="Monograptus")])
    result = task.run(note())
    assert [(t.name, t()) for t in result] == [("genus", "Monograptus"), ("age", "")]


def test_a_submit_in_the_same_reply_as_a_lookup_ends_the_conversation(tmp_path):
    both = ModelResponse(
        parts=[
            ToolCallPart("genera_lookup", {"query": "x"}, tool_call_id="l1"),
            ToolCallPart("LLMOutputs", {"genus": "G", "age": "A"}, tool_call_id="s1"),
        ],
        usage=USAGE,
    )
    task, model = build_structured(tmp_path, [both])
    with patch.object(
        type(task.pipeline.resources["genera_lookup"]), "search"
    ) as search:
        assert [t() for t in task.run(note())] == ["G", "A"]
    assert len(model.requests) == 1
    search.assert_not_called()  # the answer is in: no need to look anything up


def test_a_plain_text_reply_is_told_to_submit_the_answer(tmp_path):
    task, model = build_structured(
        tmp_path, [answer("It is Didymograptus"), submit(genus="G", age="A")]
    )
    assert [t() for t in task.run(note())] == ["G", "A"]
    nudge = str(model.requests[1][-1].parts[-1].content)
    assert "tool" in nudge


def test_a_model_that_never_submits_is_an_error(tmp_path):
    task, _ = build_structured(tmp_path, [answer("hmm")] * 10, max_tool_calls=1)
    with pytest.raises(RuntimeError, match="did not give a final answer"):
        task.run(note())


def test_structured_conversations_are_prefetched_and_reused(tmp_path):
    task, model = build_structured(
        tmp_path,
        [lookup("Didymograpus"), submit(genus="Didymograptus", age="Ordovician")],
        batch_size=2,
    )
    task.prefetch([[note()]])
    assert len(model.requests) == 2
    result = task.run(note())
    assert [t() for t in result] == ["Didymograptus", "Ordovician"]
    assert len(model.requests) == 2


def test_usage_is_recorded_for_each_turn_of_a_structured_conversation(tmp_path):
    task, _ = build_structured(
        tmp_path, [lookup("Didymograpus"), submit(genus="G", age="A")]
    )
    task.context = TaskContext(usage_file=tmp_path / "usage.yaml")
    task.run(note())
    usage = yaml.safe_load((tmp_path / "usage.yaml").read_text())["reader"]
    assert usage["total_tokens"] == 10


def test_a_lookup_can_ask_the_model_to_try_again(tmp_path):
    from pydantic_ai import ModelRetry

    pipeline, task, model = build(tmp_path, [lookup("x"), answer("fine")])
    with patch.object(
        type(pipeline.resources["genera_lookup"]),
        "search",
        side_effect=ModelRetry("search with fewer words"),
    ):
        assert task.run(note())() == "fine"
    assert "search with fewer words" in str(model.requests[1][-1].parts[-1].content)
