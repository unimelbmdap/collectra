import pytest
from typing import TypeVar
from collectra.commons import T

@pytest.fixture
def generic_type() -> TypeVar:
    return T
