from typing import TypeVar

import pytest

from collectra.commons import T


@pytest.fixture
def generic_type() -> TypeVar:
    return T
