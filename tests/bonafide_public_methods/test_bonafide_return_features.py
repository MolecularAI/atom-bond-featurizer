"""Test functions for the ``return_atom_features()`` and ``return_bond_features()`` method in the
``bonafide.bonafide`` module.
"""

from __future__ import annotations

import logging
import shutil
import warnings
from typing import TYPE_CHECKING, Any, Callable, List, Union

import pandas as pd
import pytest
from rdkit import Chem, RDLogger

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


@pytest.mark.return_features
@pytest.mark.parametrize(
    "feature_type, atom_bond_indices, output_format, reduce, temperature, ignore_invalid, "
    "_expected_output_type",
    [
        # DataFrame
        ("atom", "all", "df", False, 298.15, True, pd.DataFrame),
        ("  Bond", "All", "DF", False, 332, True, pd.DataFrame),
        ("atom", 1, "df", False, 298.15, True, pd.DataFrame),
        ("bond", 2, "df", False, 332, True, pd.DataFrame),
        ("atom", [1, 3], "df", False, 298.15, True, pd.DataFrame),
        ("bond", [0, 1], "df", False, 400, True, pd.DataFrame),
        ("atom", "all", "df", True, 298.15, True, pd.DataFrame),
        ("  Bond", "All", "DF", True, 332, True, pd.DataFrame),
        ("atom", 1, "df", True, 298.15, True, pd.DataFrame),
        ("bond", 2, "df", True, 332, True, pd.DataFrame),
        ("atom", [1, 3], "df", True, 298.15, True, pd.DataFrame),
        ("bond", [0, 1], "df", True, 400, True, pd.DataFrame),
        ("atom", "all", "df", False, 298.15, False, pd.DataFrame),
        ("  Bond", "All", "DF", False, 332, False, pd.DataFrame),
        ("atom", 1, "df", False, 298.15, False, pd.DataFrame),
        ("bond", 2, "df", False, 332, False, pd.DataFrame),
        ("atom", [1, 3], "df", False, 298.15, False, pd.DataFrame),
        ("bond", [0, 1], "df", False, 400, False, pd.DataFrame),
        ("atom", "all", "df", True, 298.15, False, pd.DataFrame),
        ("  Bond", "All", "DF", True, 332, False, pd.DataFrame),
        ("atom", 1, "df", True, 298.15, False, pd.DataFrame),
        ("bond", 2, "df", True, 332, False, pd.DataFrame),
        ("atom", [1, 3], "df", True, 298.15, False, pd.DataFrame),
        ("bond", [0, 1], "df", True, 400, False, pd.DataFrame),
        # Dictionary
        ("atom", "all", "Dict", False, 298.15, True, dict),
        ("Bond ", "All", "dict", False, 332, True, dict),
        ("atom", 1, "  DICt ", False, 298.15, True, dict),
        ("bond", 2, "dict", False, 332, True, dict),
        ("atom", [1, 3], "dict", False, 298.15, True, dict),
        ("bond", [0, 1], "dict ", False, 400, True, dict),
        ("atom", "all", "Dict", True, 298.15, True, dict),
        ("Bond ", "All", "dict", True, 332, True, dict),
        ("atom", 1, "  DICt ", True, 298.15, True, dict),
        ("bond", 2, "dict", True, 332, True, dict),
        ("atom", [1, 3], "dict", True, 298.15, True, dict),
        ("bond", [0, 1], "dict ", True, 400, True, dict),
        ("atom", "all", "Dict", False, 298.15, False, dict),
        ("Bond ", "All", "dict", False, 332, False, dict),
        ("atom", 1, "  DICt ", False, 298.15, False, dict),
        ("bond", 2, "dict", False, 332, False, dict),
        ("atom", [1, 3], "dict", False, 298.15, False, dict),
        ("bond", [0, 1], "dict ", False, 400, False, dict),
        ("atom", "all", "Dict", True, 298.15, False, dict),
        ("Bond ", "All", "dict", True, 332, False, dict),
        ("atom", 1, "  DICt ", True, 298.15, False, dict),
        ("bond", 2, "dict", True, 332, False, dict),
        ("atom", [1, 3], "dict", True, 298.15, False, dict),
        ("bond", [0, 1], "dict ", True, 400, False, dict),
        # List of mol objects or mol object (if reduce is True)
        ("atom", "all", "mol_object", False, 298.15, True, list),
        ("Bond ", "All", "MOL_OBJECT", False, 332, True, list),
        ("atom", 1, "  mol_object", False, 298.15, True, list),
        ("bond", 2, "mol_object", False, 332, True, list),
        ("atom", [1, 3], "mol_object", False, 298.15, True, list),
        ("bond", [0, 1], "mol_object", False, 400, True, list),
        ("atom", "all", "mol_object", True, 298.15, True, Chem.rdchem.Mol),
        ("Bond ", "All", "MOL_OBJECT", True, 332, True, Chem.rdchem.Mol),
        ("atom", 1, "  mol_object", True, 298.15, True, Chem.rdchem.Mol),
        ("bond", 2, "mol_object", True, 332, True, Chem.rdchem.Mol),
        ("atom", [1, 3], "mol_object", True, 298.15, True, Chem.rdchem.Mol),
        ("bond", [0, 1], "mol_object", True, 400, True, Chem.rdchem.Mol),
        ("atom", "all", "mol_object", False, 298.15, False, list),
        ("Bond ", "All", "MOL_OBJECT", False, 332, False, list),
        ("atom", 1, "  mol_object", False, 298.15, False, list),
        ("bond", 2, "mol_object", False, 332, False, list),
        ("atom", [1, 3], "mol_object", False, 298.15, False, list),
        ("bond", [0, 1], "mol_object", False, 400, False, list),
        ("atom", "all", "mol_object", True, 298.15, False, Chem.rdchem.Mol),
        ("Bond ", "All", "MOL_OBJECT", True, 332, False, Chem.rdchem.Mol),
        ("atom", 1, "  mol_object", True, 298.15, False, Chem.rdchem.Mol),
        ("bond", 2, "mol_object", True, 332, False, Chem.rdchem.Mol),
        ("atom", [1, 3], "mol_object", True, 298.15, False, Chem.rdchem.Mol),
        ("bond", [0, 1], "mol_object", True, 400, False, Chem.rdchem.Mol),
    ],
)
def test_return_features(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    fetch_data_file: Callable[[str], str],
    clean_up_logfile: Callable[[str], None],
    feature_type: str,
    atom_bond_indices: Union[str, int, List[int]],
    output_format: str,
    reduce: bool,
    temperature: float,
    ignore_invalid: bool,
    _expected_output_type: Any,
) -> None:
    """Test for the ``return_atom_features()`` and ``return_bond_features()`` method: valid inputs
    including the update of the Boltzmann weights.
    """
    # Setup featurizer
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Read input
    input_string = fetch_data_file(file_name="clopidogrel_e.sdf")
    f.read_input(
        input_value=input_string,
        namespace="irrelevant",
        input_format="file",
        read_energy=True,
        output_directory="irrelevant_out_dir",
    )
    assert f.mol_vault is not None
    assert f.mol_vault.energies_n_read is True

    # Make two dummy feature calculations
    fidx = f.list_atom_features(name="atomic_number").index.to_list()[0]
    f.featurize_atoms(atom_indices="all", feature_indices=fidx)

    fidx = f.list_bond_features(name="bond_length", origin="rdkit").index.to_list()[0]
    f.featurize_bonds(bond_indices="all", feature_indices=fidx)

    # Return features
    assert f.mol_vault.boltzmann_weights == ()

    if feature_type.strip().lower() == "atom":
        fout = f.return_atom_features(
            atom_indices=atom_bond_indices,
            output_format=output_format,
            reduce=reduce,
            temperature=temperature,
            ignore_invalid=ignore_invalid,
        )
    if feature_type.strip().lower() == "bond":
        fout = f.return_bond_features(
            bond_indices=atom_bond_indices,
            output_format=output_format,
            reduce=reduce,
            temperature=temperature,
            ignore_invalid=ignore_invalid,
        )

    assert type(fout) == _expected_output_type
    if _expected_output_type == list:
        assert all(isinstance(mol, Chem.rdchem.Mol) for mol in fout)

    # Check Boltzmann weights
    if reduce is True:
        assert len(f.mol_vault.boltzmann_weights) == 2
        assert f.mol_vault.boltzmann_weights[0] == temperature
        assert type(f.mol_vault.boltzmann_weights[1]) == list
        assert len(f.mol_vault.boltzmann_weights[1]) == f.mol_vault.size
        assert all(type(w) == float for w in f.mol_vault.boltzmann_weights[1])
    else:
        assert f.mol_vault.boltzmann_weights == ()

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)

    # Clean up
    shutil.rmtree(path="irrelevant_out_dir")
    clean_up_logfile()


@pytest.mark.return_features
@pytest.mark.parametrize("feature_type", ["atom", "bond"])
def test_return_features2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    feature_type: str,
) -> None:
    """Test for the ``return_atom_features()`` and ``return_bond_features()`` method: fails because
    no molecule read yet.
    """
    # Setup featurizer
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Return features
    if feature_type == "atom":
        with pytest.raises(ValueError, match="returning features"):
            f.return_atom_features()
    if feature_type == "bond":
        with pytest.raises(ValueError, match="returning features"):
            f.return_bond_features()

    assert f.mol_vault is None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.return_features
@pytest.mark.parametrize(
    "feature_type, atom_bond_indices, output_format, reduce, temperature, ignore_invalid, "
    "_error_type",
    [
        # Invalid inputs
        ("atom", "all", True, False, 298.15, True, TypeError),
        ("atom", "all", "df", "False", 298.15, True, TypeError),
        ("atom", "all", "df", False, {298.15}, True, TypeError),
        ("atom", "all", "df", False, 298.15, "yes", TypeError),
        ("atom", "all", "df", False, -10, True, ValueError),
        ("atom", "all", "dictionary", False, 298.15, True, ValueError),
        ("atom", -10, "df", False, 298.15, True, ValueError),
        ("atom", [0, 1, 10000000], "df", False, 298.15, True, ValueError),
        ("bond", "all", True, False, 298.15, True, TypeError),
        ("bond", "all", "df", "False", 298.15, True, TypeError),
        ("bond", "all", "df", False, {298.15}, True, TypeError),
        ("bond", "all", "df", False, 298.15, "yes", TypeError),
        ("bond", "all", "df", False, -10, True, ValueError),
        ("bond", "all", "dictionary", False, 298.15, True, ValueError),
        ("bond", -10, "df", False, 298.15, True, ValueError),
        ("bond", [0, 1, 10000000], "df", False, 298.15, True, ValueError),
        # Only valid inputs but no features calculated so far
        ("atom", "all", "df", False, 298.15, True, ValueError),
        ("bond", [0, 1], "dict", False, 298.15, True, ValueError),
    ],
)
def test_return_features3(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    feature_type: Any,
    atom_bond_indices: Any,
    output_format: Any,
    reduce: Any,
    temperature: Any,
    ignore_invalid: Any,
    _error_type: Any,
) -> None:
    """Test for the ``return_atom_features()`` and ``return_bond_features()`` method: invalid
    inputs and no features calculated.
    """
    # Setup featurizer
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Read dummy input
    f.read_input(input_value="CCO", namespace="irrelevant", input_format="smiles")
    assert f.mol_vault is not None

    # Return features
    if feature_type == "atom":
        with pytest.raises(_error_type):
            f.return_atom_features(
                atom_indices=atom_bond_indices,
                output_format=output_format,
                reduce=reduce,
                temperature=temperature,
                ignore_invalid=ignore_invalid,
            )

    if feature_type == "bond":
        with pytest.raises(_error_type):
            f.return_bond_features(
                bond_indices=atom_bond_indices,
                output_format=output_format,
                reduce=reduce,
                temperature=temperature,
                ignore_invalid=ignore_invalid,
            )

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()
