import pytest
from typing import TypeVar
from collectra.types.base import T


@pytest.fixture
def generic_type() -> TypeVar:
    return T
