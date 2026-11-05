import logging
import traceback

from rich.logging import RichHandler


class PlainTracebackHandler(RichHandler):
    """Rich log lines, with an exception shown as Python's own traceback.

    The traceback is printed whole, under the message, not wrapped and indented
    into the message column.
    """

    def emit(self, record: logging.LogRecord) -> None:
        exc_info = record.exc_info
        if not exc_info:
            return super().emit(record)
        record.exc_info, record.exc_text = None, None
        try:
            super().emit(record)
        finally:
            record.exc_info = exc_info
        self.console.print(
            "".join(traceback.format_exception(*exc_info)).rstrip(),
            markup=False,
            highlight=False,
            soft_wrap=True,
        )


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    handler = PlainTracebackHandler(show_path=False, markup=True)
    handler.setLevel(level)
    pkg_logger = logging.getLogger("collectra")
    pkg_logger.setLevel(level)
    pkg_logger.handlers.clear()
    pkg_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
