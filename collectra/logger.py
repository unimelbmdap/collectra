import logging

from rich.logging import RichHandler


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
    handler.setLevel(level)
    pkg_logger = logging.getLogger("collectra")
    pkg_logger.setLevel(level)
    pkg_logger.handlers.clear()
    pkg_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
