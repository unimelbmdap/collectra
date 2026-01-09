from typing import get_args, get_type_hints

__all__ = ["get_param_types", "get_return_type"]


def get_param_types(function) -> dict:
    param_types: dict = {
        k: v for k, v in get_type_hints(function).items() if k != "return"
    }
    if not param_types:
        return dict()
    return param_types


def get_return_type(function) -> dict:
    return_type: dict = {
        k: v for k, v in get_type_hints(function).items() if k == "return"
    }
    if not return_type:
        return dict()
    return return_type


def unpack_types(fn, unpack_fn) -> dict:
    param_types = unpack_fn(fn)
    if not param_types:
        return dict()
    for param, type_ in param_types.items():
        if get_args(type_):
            param_types[param] = tuple(
                [arg for arg in get_args(type_) if arg is not type(None)]
            )
    return param_types
