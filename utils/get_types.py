from typing import get_type_hints

__all__ = ["get_param_types", "get_return_type"]

def get_param_types(function) -> dict | None:
    param_types: dict = {k: v for k, v in get_type_hints(function).items() if k != "return"}
    if not param_types:
        return None
    return param_types

def get_return_type(function) -> type | None:
    return get_type_hints(function).get("return", None)