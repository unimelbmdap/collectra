"""Tests for collectra.llms module"""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from collectra.tasks.llms import LLM
from collectra.types import Image, Text


def test_add_text():
    mock_self = MagicMock(spec=LLM)
    result = LLM._add_text(mock_self, "test text")
    assert result == {"type": "text", "text": "test text"}


def test_add_text_empty():
    mock_self = MagicMock(spec=LLM)
    result = LLM._add_text(mock_self, "   ")
    assert result == {"type": "text", "text": "''"}


def test_add_text_empty():
    llm = LLM(name="test", model="dummy")
    result = llm._add_text("   ")
    assert result == {"type": "text", "text": "''"}


def test_add_content():
    mock_self = MagicMock(spec=LLM)

    # Test Text branch
    mock_self._add_text.return_value = {"type": "text", "text": "hello"}
    mock_text = MagicMock(spec=Text)
    mock_text.return_value = "hello"

    result = LLM._add_content(mock_self, mock_text)
    assert result == {"type": "text", "text": "hello"}

    # Test Image branch
    mock_self.image_content.return_value = {
        "type": "image",
        "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA",
    }
    mock_image = MagicMock(spec=Image)
    result = LLM._add_content(mock_self, mock_image)
    assert result == {"type": "image", "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA"}


def test_image_content():
    mock_self = MagicMock(spec=LLM)
    mock_self.llm = MagicMock()

    mock_image = MagicMock(spec=Image)
    mock_image.get_encoding.return_value = "iiVBORw0KGgoAAAANSUhEUgAAAAUA"
    mock_image.mime.return_value = "image/png"

    with patch("collectra.tasks.llms.llmloader.LLMWrapper.format") as mock_format:
        mock_format.return_value = {
            "type": "image",
            "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA",
            "mime_type": "image/png",
        }

        result = LLM.image_content(mock_self, mock_image)

        mock_format.assert_called_once_with(
            mock_self.llm,
            data_type="image",
            data={
                "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA",
                "mime_type": "image/png",
            },
        )

    assert result == {
        "type": "image",
        "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA",
        "mime_type": "image/png",
    }


def test_get_pattern_matches():
    mock_self = MagicMock(spec=LLM)
    mock_self.template = "Hello {name}"
    mock_self.preamble = "Intro"

    prompt, pattern = LLM.get_pattern_matches(mock_self)

    assert prompt == "Intro\n\nHello {name}"
    assert pattern == r"\{(.*?)\}"


def test_pre_run():
    mock_self = MagicMock(spec=LLM)
    mock_self.messages = [MagicMock(), MagicMock(), MagicMock()]  # system + extras
    mock_self.get_pattern_matches.return_value = ("Intro\n\nHello {name}", r"\{(.*?)\}")

    prompt, pattern = LLM.pre_run(mock_self)

    assert len(mock_self.messages) == 1
    assert prompt == "Intro\n\nHello {name}"
    assert pattern == r"\{(.*?)\}"


def test_check_inputs():
    mock_self = MagicMock(spec=LLM)
    mock_self.messages = [
        MagicMock(),
        MagicMock(content=[{"type": "text", "text": "hello"}]),
    ]

    LLM.check_inputs(mock_self, MagicMock(spec=Text))  # valid - should not raise

    with pytest.raises(ValueError):
        LLM.check_inputs(
            mock_self, MagicMock(spec=Text), MagicMock(spec=Text)
        )  # too many inputs

    mock_self.messages = [MagicMock()]
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


def test_replace_inputs_entities_kwarg():
    mock_self = MagicMock(spec=LLM)
    mock_self._add_text.side_effect = lambda t: {"type": "text", "text": t}

    result = LLM.replace_inputs(
        mock_self, r"\{(.*?)\}", "Result: {entities}", entities="some entity text"
    )
    assert {"type": "text", "text": "some entity text"} in result


def test_invoke():
    mock_self = MagicMock()
    mock_self.context.usage_file = None
    mock_self.parser.invoke.return_value = "hello"

    result = LLM.invoke(mock_self)

    assert result == "hello"
    mock_self.chain.invoke.assert_called_once_with(mock_self.messages)

    # empty string handling
    mock_self.parser.invoke.return_value = "''"
    result = LLM.invoke(mock_self)
    assert result == ""

    mock_self.parser.invoke.return_value = '""'
    result = LLM.invoke(mock_self)
    assert result == ""


def test_invoke_writes_usage_file(tmp_path):
    usage_file = tmp_path / "usage.yaml"
    mock_self = MagicMock()
    mock_self.name = "my_task"
    mock_self.context.usage_file = usage_file
    mock_self.parser.invoke.return_value = "response"

    with patch(
        "collectra.tasks.llms.llmloader.LLMWrapper.get_token_count"
    ) as mock_count:
        mock_count.return_value = {"prompt_tokens": 10, "completion_tokens": 5}
        LLM.invoke(mock_self)

    data = yaml.safe_load(usage_file.read_text())
    assert data == {"my_task": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_invoke_merges_usage_same_key(tmp_path):
    usage_file = tmp_path / "usage.yaml"
    usage_file.write_text(
        yaml.dump({"my_task": {"prompt_tokens": 10, "completion_tokens": 5}})
    )

    mock_self = MagicMock()
    mock_self.name = "my_task"
    mock_self.context.usage_file = usage_file
    mock_self.parser.invoke.return_value = "response"

    with patch(
        "collectra.tasks.llms.llmloader.LLMWrapper.get_token_count"
    ) as mock_count:
        mock_count.return_value = {"prompt_tokens": 3, "completion_tokens": 2}
        LLM.invoke(mock_self)

    data = yaml.safe_load(usage_file.read_text())
    assert data == {"my_task": {"prompt_tokens": 13, "completion_tokens": 7}}


def test_invoke_merges_usage_different_key(tmp_path):
    usage_file = tmp_path / "usage.yaml"
    usage_file.write_text(
        yaml.dump({"other_task": {"prompt_tokens": 10, "completion_tokens": 5}})
    )

    mock_self = MagicMock()
    mock_self.name = "my_task"
    mock_self.context.usage_file = usage_file
    mock_self.parser.invoke.return_value = "response"

    with patch(
        "collectra.tasks.llms.llmloader.LLMWrapper.get_token_count"
    ) as mock_count:
        mock_count.return_value = {"prompt_tokens": 3, "completion_tokens": 2}
        LLM.invoke(mock_self)

    data = yaml.safe_load(usage_file.read_text())
    assert data == {
        "my_task": {"prompt_tokens": 3, "completion_tokens": 2},
        "other_task": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_run_returns_text():
    mock_self = MagicMock(spec=LLM)
    mock_self.pre_run.return_value = ("Hello {name}", r"\{(.*?)\}")
    mock_self.replace_inputs.return_value = [{"type": "text", "text": "Hello Alice"}]
    mock_self.messages = [MagicMock()]
    mock_self.invoke.return_value = "some response"
    mock_self.get_output_name.return_value = "output_name"

    result = LLM.run(mock_self, MagicMock(spec=Text))

    assert isinstance(result, Text)
    assert result.name == "output_name"
    assert result.data == "some response"


def test_run_returns_none_on_exception():
    mock_self = MagicMock(spec=LLM)
    mock_self.name = "test_task"
    mock_self.pre_run.side_effect = RuntimeError("model failed")

    result = LLM.run(mock_self)

    assert result is None
