"""Test functions for the ``__init__()`` method in the ``bonafide.bonafide`` module."""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
import pytest
from rdkit import RDLogger

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


####################################
# Tests for the __init__() method. #
####################################


@pytest.mark.atom_bond_featurizer_init
@pytest.mark.parametrize(
    "log_input, error_expected, error_type",
    [
        ("valid_log.log", False, None),
        ("valid_log", False, None),
        ("_use_default_log", False, None),
        ("", True, ValueError),
        (123, True, TypeError),
        (None, True, TypeError),
        (False, True, TypeError),
    ],
)
def test___init__(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    log_input: Any,
    error_expected: bool,
    error_type: Any,
) -> None:
    """Test for the ``__init__()`` method."""
    # Error cases
    if error_expected is True:
        with pytest.raises(error_type):
            fresh_featurizer(log_file_name=log_input)
        clean_up_logfile(file_path=log_input)
        return

    # Non-error cases
    f = fresh_featurizer(log_file_name=log_input)

    assert type(getattr(f, "_feature_info")) == dict

    assert type(getattr(f, "_atom_feature_indices_2D")) == list
    assert type(getattr(f, "_bond_feature_indices_2D")) == list
    assert type(getattr(f, "_atom_feature_indices_3D")) == list
    assert type(getattr(f, "_bond_feature_indices_3D")) == list

    assert all(type(i) == int for i in f._atom_feature_indices_2D)
    assert all(type(i) == int for i in f._bond_feature_indices_2D)
    assert all(type(i) == int for i in f._atom_feature_indices_3D)
    assert all(type(i) == int for i in f._bond_feature_indices_3D)

    assert type(getattr(f, "_feature_info_df")) == pd.core.frame.DataFrame
    assert len(f._feature_info_df.columns) == 11
    assert f._feature_info_df.index.name == "INDEX"

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)

    # Clean up
    clean_up_logfile(file_path=log_input)


@pytest.mark.atom_bond_featurizer_init
def test___init__2(
    fresh_featurizer: AtomBondFeaturizer, clean_up_logfile: Callable[[str], None]
) -> None:
    """Test for the ``__init__()`` method: log file already exists."""
    # Create an empty log file
    _log_file_name = "i_already_exist.log"
    with open(file=_log_file_name, mode="w") as f:
        f.write("")

    # Test initialization
    with pytest.raises(
        FileExistsError,
        match="already exists. Remove or rename the file before running BONAFIDE with the "
        "provided log file name.",
    ):
        fresh_featurizer(log_file_name=_log_file_name)

    # Clean up
    os.remove(path=_log_file_name)
    clean_up_logfile(file_path=_log_file_name)
