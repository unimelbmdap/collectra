"""Tests for collectra.tasks.llms module"""

from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from collectra.tasks.llms import LLM
from collectra.types import Image, Text


@pytest.fixture
def llm():
    return LLM(name="test", model="dummy")


@pytest.fixture
def llm_template():
    return LLM(name="test", model="dummy", template="Hello {name}", preamble="Intro")


@pytest.fixture
def llm_run():
    return LLM(name="test", model="dummy", template="{label}", output="result")


@pytest.fixture
def img_path(tmp_path):
    from PIL import Image as ImagePil

    img = ImagePil.new("RGB", (10, 10), color=(255, 0, 0))
    path = tmp_path / "test.png"
    img.save(path)
    return path


# =============================================================================
# _add_text
# =============================================================================


def test_add_text(llm):
    assert llm._add_text("test text") == {"type": "text", "text": "test text"}


def test_add_text_strips_whitespace(llm):
    assert llm._add_text("  hello  ") == {"type": "text", "text": "hello"}


def test_add_text_empty(llm):
    assert llm._add_text("   ") == {"type": "text", "text": "''"}


# =============================================================================
# _add_content
# =============================================================================


def test_add_content_text(llm):
    text = Text("t", data="hello")
    assert llm._add_content(text) == {"type": "text", "text": "hello"}


def test_add_content_image(llm, img_path):
    image = Image("img", data=img_path)
    result = llm._add_content(image)
    assert result["type"] == "image"


# =============================================================================
# image_content
# =============================================================================


def test_image_content(llm, img_path):
    image = Image("img", data=img_path)
    result = llm.image_content(image)
    assert result == {
        "type": "image",
        "source_type": "base64",
        "mime_type": image.mime(),
        "data": image.get_encoding(),
    }


# =============================================================================
# get_pattern_matches
# =============================================================================


def test_get_pattern_matches(llm_template):
    prompt, pattern = llm_template.get_pattern_matches()
    assert prompt == "Intro\n\nHello {name}"
    assert pattern == r"\{(.*?)\}"


# =============================================================================
# pre_run
# =============================================================================


def test_pre_run(llm_template):
    llm_template.messages.append(HumanMessage(content="extra1"))
    llm_template.messages.append(HumanMessage(content="extra2"))
    prompt, pattern = llm_template.pre_run()
    assert len(llm_template.messages) == 1
    assert prompt == "Intro\n\nHello {name}"
    assert pattern == r"\{(.*?)\}"


# =============================================================================
# check_inputs
# =============================================================================


def test_check_inputs_valid(llm):
    llm.messages.append(HumanMessage(content=[{"type": "text", "text": "hello"}]))
    llm.check_inputs(Text("t", data="hello"))


def test_check_inputs_too_many_inputs(llm):
    llm.messages.append(HumanMessage(content=[{"type": "text", "text": "hello"}]))
    with pytest.raises(ValueError):
        llm.check_inputs(Text("t", data="a"), Text("t", data="b"))


def test_check_inputs_wrong_message_count(llm):
    with pytest.raises(ValueError):
        llm.check_inputs(Text("t", data="hello"))  # only system message, no human


# =============================================================================
# replace_inputs
# =============================================================================


def test_replace_inputs(llm):
    text = Text("name", data="Alice")
    result = llm.replace_inputs(r"\{(.*?)\}", "Hello {name}", text)
    assert result == [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "Alice"},
    ]


def test_replace_inputs_unknown_key(llm):
    result = llm.replace_inputs(r"\{(.*?)\}", "Hello {unknown}")
    assert any("No content provided" in r.get("text", "") for r in result)


def test_replace_inputs_entities_kwarg(llm):
    result = llm.replace_inputs(
        r"\{(.*?)\}", "Result: {entities}", entities="some entity text"
    )
    assert result == [
        {"type": "text", "text": "Result:"},
        {"type": "text", "text": "some entity text"},
    ]


# =============================================================================
# invoke
# =============================================================================


def test_invoke_returns_string(llm):
    result = llm.invoke()
    assert isinstance(result, str)
    assert len(result) > 0
    assert result == result.strip()


def test_invoke_empty_string_handling(llm):
    class _FixedChain:
        def __init__(self, val):
            self._val = val

        def invoke(self, _):
            return self._val

    llm.chain = _FixedChain("''")
    assert llm.invoke() == ""

    llm.chain = _FixedChain('""')
    assert llm.invoke() == ""


# =============================================================================
# run
# =============================================================================


def test_run_returns_text(llm_run):
    result = llm_run.run(Text("label", data="hello"))
    assert isinstance(result, Text)
    assert result.name == "result"
    assert isinstance(result.data, str) and result.data
    assert "hello" in result.data


def test_run_returns_none_on_exception(llm):
    with patch.object(llm, "pre_run", side_effect=RuntimeError("model failed")):
        result = llm.run()
    assert result is None
