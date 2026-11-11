"""Tests for collectra.llms: prompts, models, batching and usage."""

from unittest.mock import MagicMock, patch

import pytest
import yaml
from PIL import Image as PILImage
from pydantic_ai.messages import BinaryContent, CachePoint

from collectra.commons import TaskContext
from collectra.tasks.llms import LLM, LLMSetupError, model_id
from collectra.types import Image, Text
from tests.unit.scripted_llm import Scripted, prompt_parts, prompt_text

# --- building the prompt (unchanged from the LangChain version) -----------------


def test_add_text():
    mock_self = MagicMock(spec=LLM)
    result = LLM._add_text(mock_self, "test text")
    assert result == {"type": "text", "text": "test text"}


def test_empty_text_is_sent_as_two_quotes():
    assert LLM._add_text(MagicMock(spec=LLM), "  ") == {"type": "text", "text": "''"}


def test_add_content():
    mock_self = MagicMock(spec=LLM)

    # Test Text branch
    mock_self._add_text.return_value = {"type": "text", "text": "hello"}
    mock_text = MagicMock(spec=Text)
    mock_text.return_value = "hello"

    result = LLM._add_content(mock_self, mock_text)
    assert result == {"type": "text", "text": "hello"}

    # Test Image branch
    mock_self.image_content.return_value = {"type": "image", "data": b"png"}
    mock_image = MagicMock(spec=Image)
    result = LLM._add_content(mock_self, mock_image)
    assert result == {"type": "image", "data": b"png"}


def test_image_content_holds_the_encoded_image_and_its_type():
    mock_image = MagicMock(spec=Image)
    mock_image.load.return_value = b"\x89PNG"
    mock_image.mime.return_value = "image/png"
    assert LLM.image_content(MagicMock(spec=LLM), mock_image) == {
        "type": "image",
        "data": b"\x89PNG",
        "mime_type": "image/png",
    }


def test_get_pattern_matches():
    mock_self = MagicMock(spec=LLM)
    mock_self.template = "Hello {name}"
    mock_self.preamble = "Intro"

    prompt, pattern = LLM.get_pattern_matches(mock_self)

    assert prompt == "Intro\n\nHello {name}"
    assert pattern == r"\{(.*?)\}"


def test_prompt_is_an_alias_for_template():
    assert LLM("reader", model="m", prompt="Read {image}").template == "Read {image}"
    assert LLM("reader", model="m", template="Read {image}").template == "Read {image}"
    with pytest.raises(ValueError, match="both template and prompt"):
        LLM("reader", model="m", template="a", prompt="b")


def test_empty_prompt_is_reported_clearly():
    with pytest.raises(ValueError, match="has no prompt; set template:"):
        LLM("reader", model="m").get_pattern_matches()


def test_pre_run():
    mock_self = MagicMock(spec=LLM)
    mock_self.messages = ["system", [], []]  # system + extras
    mock_self.get_pattern_matches.return_value = ("Intro\n\nHello {name}", r"\{(.*?)\}")

    prompt, pattern = LLM.pre_run(mock_self)

    assert mock_self.messages == ["system"]
    assert prompt == "Intro\n\nHello {name}"
    assert pattern == r"\{(.*?)\}"


def test_check_inputs():
    mock_self = MagicMock(spec=LLM)
    mock_self.messages = ["system", [{"type": "text", "text": "hello"}]]

    LLM.check_inputs(mock_self, MagicMock(spec=Text))  # valid - should not raise

    with pytest.raises(ValueError):
        LLM.check_inputs(
            mock_self, MagicMock(spec=Text), MagicMock(spec=Text)
        )  # too many inputs

    mock_self.messages = ["system"]
    with pytest.raises(ValueError):
        LLM.check_inputs(mock_self, MagicMock(spec=Text))  # wrong message count


def test_replace_inputs():
    mock_self = MagicMock(spec=LLM)
    mock_self._add_text.side_effect = lambda t: {"type": "text", "text": t}
    mock_self._add_content.side_effect = lambda v: {"type": "text", "text": v()}

    mock_text = MagicMock(spec=Text)
    mock_text.name = "name"
    mock_text.return_value = "Alice"

    result = LLM.replace_inputs(mock_self, r"\{(.*?)\}", "Hello {name}", mock_text)
    assert {"type": "text", "text": "Alice"} in result

    result = LLM.replace_inputs(mock_self, r"\{(.*?)\}", "Hello {unknown}")
    assert any("No content provided" in r.get("text", "") for r in result)


# --- what reaches the model ----------------------------------------------------------


def task_with(model, **kwargs):
    kwargs.setdefault("template", "{text}")
    task = LLM("describe", "fake-model", **kwargs)
    task.__dict__["llm"] = model  # bypasses loading the real model
    return task


@pytest.fixture
def label_image(tmp_path):
    path = tmp_path / "label.tif"
    PILImage.new("RGB", (30, 20), "white").save(path)
    return Image(name="label", data=path)


def test_text_and_images_reach_the_model_in_template_order(label_image):
    model = Scripted(["ok"])
    task = task_with(model, template="Read {label} then {note}. And {missing}.")
    task.run(label_image, Text(name="note", data="the note"))
    parts = prompt_parts(model.requests[0])
    image = next(p for p in parts if isinstance(p, BinaryContent))
    assert [p if isinstance(p, str) else type(p).__name__ for p in parts] == [
        "Read",
        "CachePoint",
        "BinaryContent",
        "then",
        "the note",
        ". And",
        "No content provided for missing. Ignore this part.",
        ".",
    ]
    # The image goes as the task always sent it: a TIFF crop as a JPEG.
    assert image.media_type == "image/jpeg"
    assert image.data == label_image.load()


def test_the_cache_point_marks_the_fixed_text_before_the_first_image(label_image):
    model = Scripted(["ok", "ok"])
    task_with(model, template="Fields...\n{label}").run(label_image)
    parts = prompt_parts(model.requests[0])
    assert isinstance(parts[1], CachePoint) and parts[0] == "Fields..."
    # Nothing to cache when the image comes first, or with caching off.
    task_with(model, template="{label} Fields", cache=False).run(label_image)
    assert not any(isinstance(p, CachePoint) for p in prompt_parts(model.requests[1]))


def test_an_image_first_needs_no_cache_point(label_image):
    model = Scripted(["ok"])
    task_with(model, template="{label} Fields").run(label_image)
    assert not any(isinstance(p, CachePoint) for p in prompt_parts(model.requests[0]))


def test_the_system_prompt_is_sent_as_the_instructions():
    model = Scripted(["a", "b"])
    task_with(model).run(Text(name="text", data="x"))
    assert model.infos[0].instructions == "You are a helpful assistant."
    task_with(model, system="You are a curator.").run(Text(name="text", data="x"))
    assert model.infos[1].instructions == "You are a curator."


def test_the_reply_is_stripped_and_quotes_mean_empty():
    model = Scripted(["  an answer \n", "''", '""'])
    task = task_with(model)
    assert [task.run(Text(name="text", data="x"))() for _ in range(3)] == [
        "an answer",
        "",
        "",
    ]


def test_run_returns_text():
    mock_self = MagicMock(spec=LLM)
    mock_self.is_structured = False
    mock_self._prefetched = {}
    mock_self._build_messages.return_value = [{"type": "text", "text": "Hello Alice"}]
    mock_self.invoke.return_value = "some response"
    mock_self.text_output.side_effect = lambda data: Text(name="output_name", data=data)

    result = LLM.run(mock_self, MagicMock(spec=Text))

    assert isinstance(result, Text)
    assert result.name == "output_name"
    assert result.data == "some response"


def test_run_raises_when_the_model_call_fails():
    # So the pipeline records the failure and a later run tries again, rather
    # than saving no output as if the model had found nothing.
    task = task_with(Scripted([RuntimeError("model failed")]))
    with pytest.raises(RuntimeError, match="model failed"):
        task.run(Text(name="text", data="x"))


def test_runs_from_other_threads_share_one_event_loop():
    import threading

    model = Scripted(["a", "b"])
    task = task_with(model)
    results = []
    thread = threading.Thread(
        target=lambda: results.append(task.run(Text(name="text", data="x"))())
    )
    thread.start()
    thread.join()
    results.append(task.run(Text(name="text", data="y"))())
    assert results == ["a", "b"]


def test_a_run_inside_a_running_event_loop_still_works():
    import asyncio

    task = task_with(Scripted(["inside"]))

    async def main():
        return task.run(Text(name="text", data="x"))()

    assert asyncio.run(main()) == "inside"


# --- choosing the model --------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("claude-sonnet-5-5", "anthropic:claude-sonnet-5-5"),
        ("gpt-5", "openai:gpt-5"),
        ("o3-mini", "openai:o3-mini"),
        ("gemini-2.5-pro", "google:gemini-2.5-pro"),
        ("gemma-3-27b-it", "google:gemma-3-27b-it"),
        ("mistral-large-latest", "mistral:mistral-large-latest"),
        ("pixtral-large-latest", "mistral:pixtral-large-latest"),
        ("grok-4", "xai:grok-4"),
        ("anthropic:claude-opus-5-5", "anthropic:claude-opus-5-5"),
        ("ollama:qwen2.5vl:7b", "ollama:qwen2.5vl:7b"),
    ],
)
def test_model_names_keep_working_without_a_provider(name, expected):
    assert model_id(name) == expected


def test_an_unknown_bare_name_asks_for_a_provider():
    with pytest.raises(LLMSetupError, match="provider:model.*ollama:"):
        model_id("llama3")


def test_the_model_is_built_from_the_model_id():
    with patch("pydantic_ai.models.infer_model") as infer:
        model = LLM("t", "claude-sonnet-5-5", template="{x}").llm
    infer.assert_called_once_with("anthropic:claude-sonnet-5-5")
    assert model is infer.return_value


def test_a_missing_optional_dependency_names_the_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_pydantic_ai(name, *args, **kwargs):
        if name.startswith("pydantic_ai"):
            raise ModuleNotFoundError(name, name="pydantic_ai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pydantic_ai)
    with pytest.raises(ImportError, match=r"collectra\[ml\]"):
        LLM("t", "claude-sonnet-5-5", template="{x}").llm


def test_other_import_errors_are_not_hidden(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "pydantic_ai.models":
            raise ModuleNotFoundError("anthropic", name="anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(ModuleNotFoundError, match="anthropic"):
        LLM("t", "claude-sonnet-5-5", template="{x}").llm


# --- batching -------------------------------------------------------------


def entries_for(*texts):
    return [[Text(name="text", data=text)] for text in texts]


def by_text(prefix="batched"):
    return lambda text: f"{prefix} {text}"


def test_batch_size_defaults_to_one_and_is_validated():
    assert task_with(Scripted()).batch_size == 1
    assert task_with(Scripted(), batch_size=4).batch_size == 4
    with pytest.raises(ValueError, match="batch_size"):
        task_with(Scripted(), batch_size=0)


def test_prefetch_sends_at_most_batch_size_prompts_at_once():
    model = Scripted(reply=by_text(), delay=0.02)
    task = task_with(model, batch_size=2)
    task.prefetch(entries_for("a", "b", "c"))
    assert sorted(prompt_text(r) for r in model.requests) == ["a", "b", "c"]
    assert model.most_at_once == 2


def test_run_reuses_prefetched_responses_without_calling_the_model_again():
    model = Scripted(reply=by_text())
    task = task_with(model, batch_size=3)
    task.prefetch(entries_for("a", "b"))
    asked = len(model.requests)
    first, second = (task.run(*entry) for entry in entries_for("a", "b"))
    assert len(model.requests) == asked
    assert (first.data, second.data) == ("batched a", "batched b")


def test_prefetched_response_is_matched_to_its_own_inputs():
    model = Scripted(reply=by_text())
    task = task_with(model, batch_size=3)
    task.prefetch(entries_for("a", "b"))
    # Consume out of order: each response must belong to its own prompt.
    assert task.run(*entries_for("b")[0]).data == "batched b"
    assert task.run(*entries_for("a")[0]).data == "batched a"
    assert len(model.requests) == 2


def test_each_prefetched_response_is_used_once_then_the_model_is_called():
    model = Scripted(reply=by_text())
    task = task_with(model, batch_size=2)
    task.prefetch(entries_for("a"))
    assert task.run(*entries_for("a")[0]).data == "batched a"
    assert task.run(*entries_for("a")[0]).data == "batched a"
    assert len(model.requests) == 2


def test_failed_batch_items_fall_back_to_individual_calls(caplog):
    failed = set()

    def reply(text):
        if text == "b" and text not in failed:
            failed.add(text)
            return RuntimeError("boom")
        return f"answer to {text}"

    model = Scripted(reply=reply)
    task = task_with(model, batch_size=2)
    with caplog.at_level("WARNING", logger="collectra"):
        task.prefetch(entries_for("a", "b"))
    assert "will retry alone" in caplog.text
    assert task.run(*entries_for("a")[0]).data == "answer to a"
    assert task.run(*entries_for("b")[0]).data == "answer to b"
    assert [prompt_text(r) for r in model.requests].count("b") == 2


def test_prefetch_skips_entries_that_cannot_be_built(caplog):
    model = Scripted(reply=by_text())
    task = task_with(model, batch_size=2)
    with caplog.at_level("WARNING", logger="collectra"):
        task.prefetch([[], [Text(name="text", data="a"), Text(name="x", data="b")]])
        task.prefetch([[], *entries_for("a")])
    assert "could not prepare an entry" in caplog.text
    assert [prompt_text(r) for r in model.requests] == ["a"]


def test_prefetch_with_nothing_to_send_makes_no_calls():
    model = Scripted()
    task = task_with(model, batch_size=2)
    task.prefetch([])
    assert model.requests == []


def test_clear_prefetch_discards_cached_responses():
    model = Scripted(reply=by_text())
    task = task_with(model, batch_size=2)
    task.prefetch(entries_for("a"))
    task.clear_prefetch()
    task.run(*entries_for("a")[0])
    assert len(model.requests) == 2


# --- usage ------------------------------------------------------------------------------


def test_usage_is_recorded_for_the_file_being_run_not_the_one_prefetched(tmp_path):
    model = Scripted(reply=by_text())
    task = task_with(model, batch_size=2)
    task.prefetch(entries_for("a"))
    usage_file = tmp_path / "usage.yaml"
    task.context = TaskContext(usage_file=usage_file)
    task.run(*entries_for("a")[0])
    assert yaml.safe_load(usage_file.read_text()) == {
        "describe": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
    }


def test_usage_adds_up_across_runs_and_keeps_other_tasks(tmp_path):
    usage_file = tmp_path / "usage.yaml"
    usage_file.write_text(yaml.safe_dump({"other": {"input_tokens": 7}}))
    task = task_with(Scripted(["a", "b"]))
    task.context = TaskContext(usage_file=usage_file)
    task.run(Text(name="text", data="x"))
    task.run(Text(name="text", data="y"))
    data = yaml.safe_load(usage_file.read_text())
    assert data["other"] == {"input_tokens": 7}
    assert data["describe"]["input_tokens"] == 6
    assert data["describe"]["total_tokens"] == 10


def test_no_usage_file_means_nothing_is_written(tmp_path):
    task = task_with(Scripted(["a"]))
    task.run(Text(name="text", data="x"))
    assert list(tmp_path.iterdir()) == []


# --- settings -------------------------------------------------------------------


def test_temperature_is_not_set_unless_asked_for():
    """Some models reject ``temperature``; the model's own default is the default."""
    assert LLM("t", "m", template="{x}").temperature is None


def test_an_explicit_temperature_is_kept():
    assert LLM("t", "m", template="{x}", temperature=0.2).temperature == 0.2
    assert LLM("t", "m", template="{x}", temperature=0).temperature == 0


def test_the_model_settings_hold_only_what_was_asked_for():
    assert LLM("t", "m", template="{x}").model_settings() == {"cache": True}
    assert LLM(
        "t", "m", template="{x}", temperature=0.3, max_tokens=50, cache=False
    ).model_settings() == {"temperature": 0.3, "max_tokens": 50, "cache": False}


def test_the_settings_are_sent_with_each_request():
    model = Scripted(["a"])
    task_with(model, temperature=0, max_tokens=50).run(Text(name="text", data="x"))
    settings = model.infos[0].model_settings
    assert settings["temperature"] == 0 and settings["max_tokens"] == 50
