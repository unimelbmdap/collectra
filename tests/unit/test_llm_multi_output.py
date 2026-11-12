"""An LLM with several output nodes answers in structured form, one Text per node."""

from unittest.mock import patch

import pytest
import yaml
from pydantic import BaseModel

from collectra import Pipeline
from collectra.commons import TaskContext
from collectra.tasks.base import Task
from collectra.tasks.llms import LLM
from collectra.types.images import Image
from collectra.types.texts import Text
from tests.unit.test_pipeline_stages import files  # noqa: F401
from tests.unit.scripted_llm import Scripted, prompt_text
from utils.get_types import get_return_type, unpack_types

THIS = "tests.unit.test_llm_multi_output"
OUTPUTS = ["registration_number_draft", "genus_draft", "age_draft"]
LABEL = {
    "registration_number_draft": "P 1234",
    "genus_draft": "Calamites",
    "age_draft": "Carboniferous",
}


def FakeLLM(reply=lambda text: dict(LABEL), supported=True):
    """A model that answers each prompt with ``reply(prompt text)``."""
    return Scripted(reply=reply, supports_tools=supported)


def reader(llm=None, outputs=OUTPUTS, **kwargs):
    task = LLM(
        "reader", "m", template="{label}", input="label", output=outputs, **kwargs
    )
    task.__dict__["llm"] = llm or FakeLLM()
    return task


def label(text="Calamites sp. P 1234"):
    return Text(name="label", data=text)


# --- which tasks are structured ---------------------------------------------


def test_output_names_accept_a_name_a_list_or_nothing():
    assert LLM("t", "m", template="x", output="one").output_names == ["one"]
    assert LLM("t", "m", template="x", output=["a", "b"]).output_names == ["a", "b"]
    assert LLM("t", "m", template="x").output_names == []


def test_only_several_outputs_make_the_task_structured():
    assert not LLM("t", "m", template="x").is_structured
    assert not LLM("t", "m", template="x", output="one").is_structured
    assert not LLM("t", "m", template="x", output=["one"]).is_structured
    assert LLM("t", "m", template="x", output=["a", "b"]).is_structured


def test_the_same_output_cannot_be_listed_twice():
    with pytest.raises(ValueError, match="more than once"):
        LLM("t", "m", template="x", output=["a", "a"])


# --- the schema built from the node names ------------------------------------


def test_the_schema_has_one_string_property_per_output_node():
    schema = reader().output_model
    assert issubclass(schema, BaseModel)
    properties = schema.model_json_schema()["properties"]
    assert list(properties) == OUTPUTS
    assert all(p["type"] == "string" for p in properties.values())
    assert schema.model_json_schema().get("required", []) == []


def test_awkward_node_names_still_work_as_properties():
    names = ["json", "copy", "_private", "a-b", "9lives", "model_config"]
    model = reader(outputs=names).output_model
    assert list(model.model_json_schema()["properties"]) == names
    parsed = model.model_validate({"json": "j", "a-b": "ab"})
    assert [getattr(parsed, f"field_{i}") for i in range(len(names))] == [
        "j",
        "",
        "",
        "ab",
        "",
        "",
    ]


def test_the_schema_tells_the_model_to_leave_absent_fields_empty():
    assert "empty" in reader().output_model.__doc__


@pytest.mark.parametrize(
    "value, expected", [(None, ""), ("  x  ", "  x  "), (5, "5"), (1.5, "1.5")]
)
def test_values_are_made_into_strings(value, expected):
    model = reader().output_model
    assert model.model_validate({"genus_draft": value}).field_1 == expected


# --- running ----------------------------------------------------------------


def test_run_returns_one_text_per_output_node_in_order():
    result = reader().run(label())
    assert [t.name for t in result] == OUTPUTS
    assert [t() for t in result] == [LABEL[n] for n in OUTPUTS]
    assert all(isinstance(t, Text) for t in result)


def test_the_answer_is_asked_for_as_a_tool_with_one_field_per_node():
    llm = FakeLLM()
    task = reader(llm)
    task.run(label())
    task.run(label())
    (tool,) = llm.infos[0].output_tools
    assert tool.name == "LLMOutputs"
    assert list(tool.parameters_json_schema["properties"]) == OUTPUTS
    assert not llm.infos[0].allow_text_output
    assert task.agent is task.agent  # built once, reused


def test_missing_fields_become_empty_strings():
    result = reader(FakeLLM(lambda text: {"genus_draft": "Calamites"})).run(label())
    assert [t() for t in result] == ["", "Calamites", ""]


def test_nothing_found_still_gives_every_node_an_empty_text():
    result = reader(FakeLLM(lambda text: {})).run(label())
    assert [t.name for t in result] == OUTPUTS
    assert [t() for t in result] == ["", "", ""]


def test_a_single_output_is_still_free_text():
    free = Scripted(["plain answer"])
    task = LLM("t", "m", template="{label}", output=["only"])
    task.__dict__["llm"] = free
    result = task.run(label())
    assert isinstance(result, Text) and result() == "plain answer"
    assert result.name == "only"
    assert free.infos[0].output_tools == []


def garbage():
    """A reply whose answer is not an object of fields."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    return ModelResponse(parts=[ToolCallPart("LLMOutputs", "not json at all")])


def test_a_reply_the_schema_rejects_is_sent_back_for_the_model_to_correct():
    llm = Scripted([garbage(), dict(LABEL)])
    assert [t() for t in reader(llm).run(label())] == [LABEL[n] for n in OUTPUTS]
    assert len(llm.requests) == 2


def test_a_reply_the_schema_keeps_rejecting_is_an_error_rather_than_empty_output():
    with pytest.raises(RuntimeError, match="reader.*did not give a final answer"):
        reader(Scripted(reply=lambda text: garbage())).run(label())


def test_a_model_without_tool_calling_raises_a_clear_error():
    task = reader(FakeLLM(supported=False))
    with pytest.raises(Exception, match=r"reader.*tool"):
        task.run(label())


def test_that_error_is_not_swallowed_per_page():
    from collectra.tasks.llms import StructuredOutputUnsupported

    with pytest.raises(StructuredOutputUnsupported):
        reader(FakeLLM(supported=False)).run(label())


# --- batching and usage -------------------------------------------------------


def entries(*texts):
    return [[Text(name="label", data=t)] for t in texts]


def test_batches_use_the_structured_answer_and_runs_reuse_the_parsed_replies():
    llm = Scripted(reply=lambda text: {"genus_draft": text.upper()}, delay=0.02)
    task = reader(llm, batch_size=2)
    task.prefetch(entries("a", "b", "c"))
    assert llm.most_at_once == 2 and len(llm.requests) == 3
    first = task.run(*entries("a")[0])
    assert [t() for t in first] == ["", "A", ""]
    assert len(llm.requests) == 3


def test_an_unreadable_batch_reply_is_retried_alone():
    def reply(text):
        return garbage() if text == "bad" else {"genus_draft": 5}

    llm = Scripted(reply=reply)
    task = reader(llm, batch_size=2)
    task.prefetch(entries("good", "bad"))
    asked = len(llm.requests)
    assert [t() for t in task.run(*entries("good")[0])][1] == "5"
    assert len(llm.requests) == asked
    with pytest.raises(RuntimeError):  # still unreadable when asked alone
        task.run(*entries("bad")[0])
    assert len(llm.requests) > asked
    assert {prompt_text(r) for r in llm.requests[asked:]} == {"bad"}


def test_usage_is_recorded_for_a_structured_answer(tmp_path):
    task = reader()
    task.context = TaskContext(usage_file=tmp_path / "usage.yaml")
    task.run(label())
    usage = yaml.safe_load((tmp_path / "usage.yaml").read_text())["reader"]
    assert (usage["input_tokens"], usage["output_tokens"]) == (3, 2)


# --- the graph and the saved results ------------------------------------------


class Describe(Task):
    def run(self, image: Image) -> Text:
        return Text(name=self.get_output_name(), data="Calamites sp. P 1234")


def label_pipeline(tmp_path, **llm):
    return Pipeline(
        "Lab",
        "lab",
        "0.1.0",
        path=str(tmp_path),
        first={"type": f"{THIS}.Describe", "input": ["page"], "output": ["label"]},
        reader={
            "type": "collectra.LLM",
            "model": "m",
            "template": "{label}",
            "input": ["label"],
            "output": OUTPUTS,
            **llm,
        },
    )


def folder(tmp_path):
    return tmp_path / "input" / "a.lab"


def saved(tmp_path, key):
    data = yaml.safe_load((folder(tmp_path) / "results.yaml").read_text())[key]
    return data if isinstance(data, list) else [data]


def test_every_output_node_exists_and_is_a_text_node(tmp_path):
    pipe = label_pipeline(tmp_path)
    pipe.connect()
    task = pipe.node_manager.resolve_node("reader")
    children = pipe.node_manager.get_children(task)
    assert [c.name for c in children] == OUTPUTS
    assert all(c.check_type(Text) for c in children)


def test_return_annotations_with_lists_inside_unions_name_the_node_type():
    class Both:
        def run(self, x: Text) -> Text | list[Text] | None: ...

    class OnlyList:
        def run(self, x: Text) -> list[Text] | None: ...

    assert unpack_types(Both().run, get_return_type) == {"return": (Text,)}
    assert unpack_types(OnlyList().run, get_return_type) == {"return": (Text,)}


def test_the_pipeline_fills_every_output_node(tmp_path, files):
    with patch.object(LLM, "llm", property(lambda self: FakeLLM())):
        label_pipeline(tmp_path).run(files=files[:1])
    for node in OUTPUTS:
        (item,) = saved(tmp_path, node)
        assert item["id"] == node
        assert item["data"] == LABEL[node]
        assert item["parents"] == "label"


def test_missing_values_are_saved_as_empty_text(tmp_path, files):
    fake = FakeLLM(lambda text: {"genus_draft": "Calamites"})
    with patch.object(LLM, "llm", property(lambda self: fake)):
        label_pipeline(tmp_path).run(files=files[:1])
    assert [saved(tmp_path, n)[0]["data"] for n in OUTPUTS] == ["", "Calamites", ""]


def test_a_second_run_skips_the_task_because_every_node_has_output(tmp_path, files):
    fake = FakeLLM(lambda text: {})
    with patch.object(LLM, "llm", property(lambda self: fake)):
        pipe = label_pipeline(tmp_path)
        pipe.run(files=files[:1])
        asked = len(fake.requests)
        pipe.run("reader", single=True, files=[str(folder(tmp_path))])
    assert len(fake.requests) == asked  # nothing was sent again


def test_files_are_written_only_for_nodes_with_text(tmp_path, files):
    fake = FakeLLM(
        lambda text: {"genus_draft": "Calamites", "age_draft": "Carboniferous"}
    )
    with patch.object(LLM, "llm", property(lambda self: fake)):
        label_pipeline(tmp_path, output_extension="md").run(files=files[:1])
    assert (
        folder(tmp_path) / "genus_draft" / "genus_draft.md"
    ).read_text() == "Calamites"
    assert (
        folder(tmp_path) / "age_draft" / "age_draft.md"
    ).read_text() == "Carboniferous"
    assert not (folder(tmp_path) / "registration_number_draft").exists()
    empty = saved(tmp_path, "registration_number_draft")[0]
    assert empty["data"] == "" and "path" not in empty
    assert saved(tmp_path, "genus_draft")[0]["path"] == "genus_draft/genus_draft.md"


def test_empty_text_is_never_written_to_a_file(tmp_path):
    text = Text(name="n", id="n", data="")
    text._file_extension = "md"
    text.finalise(tmp_path, "node")
    assert not (tmp_path / "node").exists()
    assert text.path is None


def test_force_replaces_all_the_nodes(tmp_path, files):
    first = FakeLLM(lambda text: {"genus_draft": "Old"})
    with patch.object(LLM, "llm", property(lambda self: first)):
        pipe = label_pipeline(tmp_path)
        pipe.run(files=files[:1])
    second = FakeLLM(lambda text: {"genus_draft": "New", "age_draft": "Devonian"})
    with patch.object(LLM, "llm", property(lambda self: second)):
        label_pipeline(tmp_path).run(
            "reader", single=True, files=[str(folder(tmp_path))], force=True
        )
    assert [saved(tmp_path, n)[0]["data"] for n in OUTPUTS] == ["", "New", "Devonian"]
    assert all(len(saved(tmp_path, n)) == 1 for n in OUTPUTS)


def test_text_output_can_name_the_node_it_belongs_to():
    task = reader()
    assert task.text_output("x", name="genus_draft").name == "genus_draft"
    assert task.text_output("x").name == "registration_number_draft"
