"""Tests for collectra.tasks.irn_resolvers pure functions"""

import math

import pandas as pd

from collectra.tasks.irn_resolvers import (
    _is_missing,
    _SafeDict,
    normalise_column_name,
    render_card,
)

# =============================================================================
# _SafeDict
# =============================================================================


def test_safe_dict_missing_key_returns_empty():
    d = _SafeDict({"a": "1"})
    assert d["missing"] == ""


def test_safe_dict_present_key_returns_value():
    d = _SafeDict({"a": "hello"})
    assert d["a"] == "hello"


# =============================================================================
# normalise_column_name
# =============================================================================


def test_normalise_basic_snake_case():
    assert normalise_column_name("Hello World") == "hello_world"


def test_normalise_strips_trailing_parenthetical():
    assert normalise_column_name("Column Name (optional)") == "column_name"


def test_normalise_strips_trailing_colon():
    assert normalise_column_name("Label:") == "label"


def test_normalise_replaces_slash_with_underscore():
    assert normalise_column_name("city/town") == "city_town"


def test_normalise_drops_special_characters():
    assert normalise_column_name("col-name!") == "colname"


def test_normalise_collapses_multiple_underscores():
    assert normalise_column_name("col  name") == "col_name"


def test_normalise_strips_leading_trailing_underscores():
    assert normalise_column_name("  name  ") == "name"


def test_normalise_numeric_input():
    assert normalise_column_name("42") == "42"


# =============================================================================
# _is_missing
# =============================================================================


def test_is_missing_none():
    assert _is_missing(None) is True


def test_is_missing_float_nan():
    assert _is_missing(float("nan")) is True


def test_is_missing_math_nan():
    assert _is_missing(math.nan) is True


def test_is_missing_pandas_na():
    assert _is_missing(pd.NA) is True


def test_is_missing_pandas_nat():
    assert _is_missing(pd.NaT) is True


def test_is_missing_regular_string():
    assert _is_missing("hello") is False


def test_is_missing_zero():
    assert _is_missing(0) is False


def test_is_missing_empty_string():
    assert _is_missing("") is False


# =============================================================================
# render_card
# =============================================================================


def test_render_card_basic():
    result = render_card("{name} | {city}", {"name": "Alice", "city": "Melbourne"})
    assert result == "Alice | Melbourne"


def test_render_card_none_value_collapsed():
    result = render_card("{name} | {city}", {"name": "Alice", "city": None})
    assert result == "Alice"


def test_render_card_nan_value_collapsed():
    result = render_card("{a} | {b}", {"a": "x", "b": float("nan")})
    assert result == "x"


def test_render_card_consecutive_separators_collapsed():
    result = render_card("{a} | {b} | {c}", {"a": "x", "b": None, "c": "z"})
    assert result == "x | z"


def test_render_card_missing_key_returns_empty():
    result = render_card("{name} | {unknown}", {"name": "Alice"})
    assert result == "Alice"


def test_render_card_all_missing_returns_empty():
    result = render_card("{a} | {b}", {"a": None, "b": None})
    assert result == ""


def test_render_card_leading_separator_stripped():
    result = render_card("{a} | {b}", {"a": None, "b": "y"})
    assert not result.startswith("|")
    assert "y" in result


def test_render_card_comma_separator():
    result = render_card("{a}, {b}, {c}", {"a": "x", "b": None, "c": "z"})
    assert result == "x, z"
