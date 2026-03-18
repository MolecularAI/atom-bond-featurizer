"""Test functions for the ``bonafide.utils.logging_format`` module."""

import logging

import pytest

from bonafide.utils.logging_format import IndentationFormatter


def _make_log_record(msg: str) -> logging.LogRecord:
    """Create a logging.LogRecord for testing."""
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


##################################
# Tests for the format() method. #
##################################


@pytest.mark.format
def test_format() -> None:
    """Test for the ``format()`` method: single-line log message."""
    fmt = IndentationFormatter(fmt="%(levelname)s: ")
    record = _make_log_record("A single line message")
    formatted = fmt.format(record)
    assert type(formatted) == str
    assert formatted.startswith("INFO: ")
    assert "A single line message" in formatted
    assert "\n" not in formatted


@pytest.mark.format
def test_format2() -> None:
    """Test for the ``format()`` method: multi-line log message."""
    fmt = IndentationFormatter(fmt="%(levelname)s: ")
    record = _make_log_record("First line\nSecond line\nThird line")
    formatted = fmt.format(record)
    assert type(formatted) == str

    lines = formatted.splitlines()
    assert lines[0].startswith("INFO: ")
    assert len(lines) == 3

    # All continuation lines should be indented to align with the message
    indent = " " * len("INFO: ")
    assert lines[1].startswith(indent)
    assert lines[2].startswith(indent)
    assert "First line" in lines[0]
    assert "Second line" in lines[1]


@pytest.mark.format
def test_format3() -> None:
    """Test for the ``format()`` method: wrapping of long message."""
    _max_length = 50
    fmt = IndentationFormatter(fmt="%(levelname)s: ", max_line_length=_max_length)
    long_message = "This is a very long message that should be wrapped into multiple lines so that no single line exceeds the maximum line length specified."
    record = _make_log_record(long_message)
    formatted = fmt.format(record)
    lines = formatted.splitlines()

    for line in lines:
        assert len(line) <= _max_length


@pytest.mark.format
def test_format4() -> None:
    """Test for the ``format()`` method: ensure that empty lines in the message are preserved and
    indented correctly.
    """
    fmt = IndentationFormatter(fmt="%(levelname)s: ")
    msg = "First line\n\nThird line"
    record = _make_log_record(msg)
    formatted = fmt.format(record)
    lines = formatted.splitlines()

    assert lines[0].startswith("INFO: ")
    assert lines[1] == ""  # Empty line preserved
    indent = " " * len("INFO: ")
    assert lines[2].startswith(indent)
    assert "Third line" in lines[2]


@pytest.mark.format
def test_format5() -> None:
    """Test for the ``format()`` method: correct formatting of multiple lines."""
    _max_length = 60
    fmt = IndentationFormatter(fmt="%(levelname)s: ", max_line_length=_max_length)
    msg = (
        "First line is very long and should be wrapped accordingly.\n"
        "Second line is also quite long and will need wrapping.\n"
        "Short"
    )
    record = _make_log_record(msg)
    formatted = fmt.format(record)
    lines = formatted.splitlines()
    indent = " " * len("INFO: ")

    # First line(s)
    assert lines[0].startswith("INFO: ")
    for line in lines[1:]:
        if line.strip():
            assert line.startswith(indent)

    # No line exceeds _max_length
    for line in lines:
        assert len(line) <= _max_length
