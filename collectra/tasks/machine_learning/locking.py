"""Small synchronization helpers without importing an ML backend."""

from __future__ import annotations

from functools import wraps
from threading import Lock


def threading_locked():
    """Serialize calls to a method, matching the backend lock decorator."""
    lock = Lock()

    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)

        return wrapped

    return decorate
