"""Tests for collectra.llms module"""

import pytest
from unittest.mock import MagicMock, patch
from collectra.tasks.llms import LLM
from collectra.types import Text, Image


def test_add_text():
    mock_self = MagicMock(spec=LLM)
    result = LLM._add_text(mock_self, "test text")
    assert result == {"type": "text", "text": "test text"}


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
        MagicMock(content=[{"type": "text", "text": "hello"}])
    ]

    LLM.check_inputs(mock_self, MagicMock(spec=Text))  # valid - should not raise

    with pytest.raises(ValueError):
        LLM.check_inputs(mock_self, MagicMock(spec=Text), MagicMock(spec=Text))  # too many inputs

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