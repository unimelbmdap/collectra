import logging

from collectra.logger import get_logger, setup_logging


def test_errors_log_a_standard_python_traceback(capsys):
    setup_logging()
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("collectra.test").error("Failed", exc_info=True)
    output = capsys.readouterr().out
    assert "Traceback (most recent call last):" in output
    assert "ValueError: boom" in output
    assert "╭" not in output


def test_setup_logging_replaces_earlier_handlers():
    setup_logging()
    setup_logging(verbose=True)
    package = logging.getLogger("collectra")
    assert len(package.handlers) == 1
    assert package.level == logging.DEBUG
    setup_logging()


def test_the_traceback_is_not_wrapped_to_the_terminal_width(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "40")
    setup_logging()
    long_message = "x" * 120
    try:
        raise ValueError(long_message)
    except ValueError:
        get_logger("collectra.test").error("Failed", exc_info=True)
    lines = capsys.readouterr().out.splitlines()
    assert f"ValueError: {long_message}" in lines
    assert any(line == "Traceback (most recent call last):" for line in lines)


def test_a_message_without_an_exception_is_logged_as_usual(capsys):
    setup_logging()
    get_logger("collectra.test").error("Plain failure")
    output = capsys.readouterr().out
    assert "Plain failure" in output
    assert "Traceback" not in output
