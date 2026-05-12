"""Tests for collectra.types.texts module"""

from pathlib import Path

import pytest

from collectra.types.texts import Text, unmark, unmark_element

# =============================================================================
# unmark_element
# =============================================================================


def test_unmark_element_text_only():
    from xml.etree.ElementTree import fromstring

    el = fromstring("<p>Hello</p>")
    assert unmark_element(el) == "Hello"


def test_unmark_element_with_tail():
    from xml.etree.ElementTree import fromstring

    el = fromstring("<root><p>Hello</p> world</root>")
    assert "Hello" in unmark_element(el)
    assert "world" in unmark_element(el)


def test_unmark_element_nested():
    from xml.etree.ElementTree import fromstring

    el = fromstring("<root><p>first</p><p>second</p></root>")
    result = unmark_element(el)
    assert "first" in result
    assert "second" in result


def test_unmark_element_with_stream():
    from io import StringIO
    from xml.etree.ElementTree import fromstring

    el = fromstring("<p>streamed</p>")
    stream = StringIO()
    result = unmark_element(el, stream)
    assert result == "streamed"


# =============================================================================
# unmark
# =============================================================================


def test_unmark_strips_markdown_bold():
    result = unmark("**bold**")
    assert "bold" in result
    assert "**" not in result


def test_unmark_strips_markdown_header():
    result = unmark("# Header")
    assert "Header" in result
    assert "#" not in result


def test_unmark_plain_text_unchanged():
    result = unmark("just plain text")
    assert "just plain text" in result


# =============================================================================
# Text construction
# =============================================================================


def test_text_plain_string():
    t = Text("t", data="hello")
    assert t() == "hello"


def test_text_empty_string():
    t = Text("t", data="")
    assert t() == ""


def test_text_loads_from_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("file content")
    t = Text("t", data=str(f))
    assert t() == "file content"


def test_text_loads_from_path_object(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("path content")
    t = Text("t", data=f)
    assert t() == "path content"


def test_text_nonexistent_path_kept_as_string(tmp_path):
    missing = str(tmp_path / "no_such_file.txt")
    t = Text("t", data=missing)
    assert t() == missing


# =============================================================================
# Text.evaluate
# =============================================================================


def test_evaluate_identical_texts():
    a = Text("a", data="hello world")
    b = Text("b", data="hello world")
    assert a.evaluate(b) == pytest.approx(1.0)


def test_evaluate_completely_different():
    a = Text("a", data="aaa")
    b = Text("b", data="zzz")
    assert a.evaluate(b) == pytest.approx(0.0)


def test_evaluate_partial_match():
    a = Text("a", data="hello world")
    b = Text("b", data="hello earth")
    ratio = a.evaluate(b)
    assert 0.0 < ratio < 1.0


def test_evaluate_invalid_type_raises():
    t = Text("t", data="hello")
    with pytest.raises(ValueError):
        t.evaluate("not a Text")
