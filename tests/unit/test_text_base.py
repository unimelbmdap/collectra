"""Tests for collectra.text.base module"""

from collectra.text.base import Text

# =============================================================================
# Text
# =============================================================================


def test_content_stored():
    t = Text("hello")
    assert t.content == "hello"


def test_str_returns_content():
    t = Text("hello world")
    assert str(t) == "hello world"
