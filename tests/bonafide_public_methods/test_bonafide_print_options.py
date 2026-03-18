"""Test functions for the ``print_options()`` method in the ``bonafide.bonafide`` module."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable

import pytest
from rdkit import RDLogger

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


@pytest.mark.print_options
@pytest.mark.parametrize(
    "origin, error_expected, error_type",
    [
        # Valid cases
        ("alfabet", False, None),
        (["rdkit"], False, None),
        (["RDkit", "multiWFN", "Psi4"], False, None),
        ([], False, None),
        ("_use_default", False, None),
        (None, False, None),
        # Invalid cases
        (False, True, TypeError),
        (123, True, TypeError),
        (["rdkit", "i_dont_exist", "multiwfn"], True, ValueError),
        ("why_am_I_here", True, ValueError),
        (["i_want_to_go_HOME"], True, ValueError),
    ],
)
def test_print_options(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    origin: Any,
    error_expected: bool,
    error_type: Any,
) -> None:
    """Test for the ``print_options()`` method."""
    f = fresh_featurizer()

    # Error cases
    if error_expected is True:
        with pytest.raises(error_type, match="Invalid input to 'origin':"):
            f.print_options(origin=origin)

        # Check no output is printed to stdout
        captured = capsys.readouterr()
        assert captured.out == ""

        # Check logs
        assert len(caplog.records) > 0
        assert any(record.levelno == logging.ERROR for record in caplog.records)

        # Clean up
        clean_up_logfile()
        return

    # Valid cases
    if origin == "_use_default":
        f.print_options()
    else:
        f.print_options(origin=origin)

    # Check printed output
    captured = capsys.readouterr()
    assert "Default configuration settings at" in captured.out
    if type(origin) == str and origin != "_use_default":
        origin = [origin]
    if type(origin) == list:
        for o in origin:
            assert o.lower() in captured.out

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)

    # Clean up
    clean_up_logfile()
