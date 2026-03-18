"""Formatting of logging messages for consistent indentation and line length."""

import logging
import textwrap
from typing import Literal, Optional


class IndentationFormatter(logging.Formatter):
    """Logging formatter that indents continuation lines to align with the start of the message.

    Parameters
    ----------
    fmt : Optional[str], optional
        The format string for the log message, by default ``None``.
    datefmt : Optional[str], optional
        The format string for the date/time, by default ``None``.
    style : str, optional
        The style of the format string, by default ``"%"``.
    max_line_length : int, optional
        The maximum line length for the formatted message, by default ``150``.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: Literal["%", "{", "$"] = "%",
        max_line_length: int = 150,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.max_line_length = max_line_length

    def format(self, record: logging.LogRecord) -> str:
        """Format logging records.

        Each logical line (between pre-existing line breaks) is wrapped individually.
        All continuation lines are indented to align with the start of the message.

        Parameters
        ----------
        record : logging.LogRecord
            The logging record to format.

        Returns
        -------
        str
            The formatted logging message with indented continuation lines.
        """
        original_message = str(record.msg)

        # Get indentation
        record.msg = ""
        prefix = super().format(record)
        indentation = " " * len(prefix)

        # Split original message by existing newlines
        logical_lines = original_message.split("\n")
        wrapped_message_lines = []

        for idx, logical_line in enumerate(logical_lines):
            # Wrap lines and introduce new line breaks if necessary
            wrapped = textwrap.wrap(
                logical_line,
                width=self.max_line_length - len(prefix),
                break_long_words=False,
                break_on_hyphens=False,
            )
            if wrapped:
                # Apply prefix only to the first visual/logical line
                if idx == 0:
                    wrapped_message_lines.append(prefix + wrapped[0])
                    for extra_line in wrapped[1:]:
                        wrapped_message_lines.append(extra_line)
                else:
                    for w in wrapped:
                        wrapped_message_lines.append(w)

            # Empty lines
            else:
                wrapped_message_lines.append("")

        # Indent all subsequent lines (created by wrap and explicit newlines)
        for idx in range(1, len(wrapped_message_lines)):
            if wrapped_message_lines[idx]:  # don't indent empty lines
                wrapped_message_lines[idx] = indentation + wrapped_message_lines[idx]

        return "\n".join(wrapped_message_lines)
