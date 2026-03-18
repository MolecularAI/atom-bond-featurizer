"""Test functions for the ``set_charge()`` method in the ``bonafide.bonafide`` module."""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable

import pytest
from rdkit import RDLogger

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


@pytest.mark.set_charge
@pytest.mark.parametrize(
    "input_string, input_format, charge, _error_type",
    [
        ("spiro_mol_e.xyz", "file", 7, None),
        ("Cr_III_cation-conf_01.sdf", "file", -7, None),
        ("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O", "smiles", 2, None),
        ("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O", "smiles", None, TypeError),
        ("Cr_III_cation-conf_01.sdf", "file", 1.5, TypeError),
    ],
)
def test_set_charge(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    input_string: str,
    input_format: str,
    charge: Any,
    _error_type: Any,
):
    """Test for the ``set_charge()`` method: valid and invalid examples."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    if input_format == "file":
        input_string = fetch_data_file(file_name=input_string)

    f.read_input(
        input_value=input_string,
        namespace="irrelevant",
        input_format=input_format,
        output_directory="irrelevant_out_dir",
    )

    # Set charge
    _init_charge = f.mol_vault.charge
    if input_format == "file":
        assert f.mol_vault.charge is None
    else:
        assert f.mol_vault.charge is not None
        assert type(f.mol_vault.charge) == int

    # Valid examples
    if _error_type is None:
        f.set_charge(charge=charge)
        assert f.mol_vault.charge != _init_charge
        assert f.mol_vault.charge == charge

        # Change charge again
        f.set_charge(charge=111)
        assert f.mol_vault.charge == 111

        # Check logs
        assert len(caplog.records) > 0
        assert all(record.levelno == logging.INFO for record in caplog.records)

    # Invalid examples
    else:
        with pytest.raises(_error_type, match="Invalid input to 'charge'"):
            f.set_charge(charge=charge)
        assert f.mol_vault.charge == _init_charge
        assert f.mol_vault.charge != charge

        # Check logs
        assert len(caplog.records) > 0
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    os.rmdir(path="irrelevant_out_dir")
    clean_up_logfile()


@pytest.mark.set_charge
def test_set_charge2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
):
    """Test for the ``set_charge()`` method: fails because no molecule read yet."""
    f = fresh_featurizer()
    assert f.mol_vault is None

    with pytest.raises(ValueError, match="setting the charge"):
        f.set_charge(charge=0)
    assert f.mol_vault is None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()
