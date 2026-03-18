"""Test functions for the ``clear_atom_feature_cache()`` and ``clear_bond_feature_cache()`` method
in the ``bonafide.bonafide`` module.
"""

from __future__ import annotations

import logging
import shutil
import warnings
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import pytest
from rdkit import RDLogger

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


@pytest.mark.clear_feature_cache
@pytest.mark.parametrize(
    "feature_type, origin",
    [
        ("Atom", None),
        ("bond ", None),
        ("Atom", "rdkit"),
        ("bond", "RDKit"),
        ("Atom", ["RDKIT"]),
        ("bond", ["rdkit"]),
    ],
)
def test_clear_feature_cache(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    fetch_data_file: Callable[[str], str],
    clean_up_logfile: Callable[[str], None],
    feature_type: str,
    origin: Optional[Union[str, List[str]]],
) -> None:
    """Test for the ``clear_atom_feature_cache()`` and ``clear_bond_feature_cache()``
    method: valid inputs.
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
        output_directory="irrelevant_out_dir",
    )
    assert f.mol_vault is not None

    # Make two dummy feature calculations
    fidx = f.list_atom_features(name="atomic_number").index.to_list()[0]
    f.featurize_atoms(atom_indices="all", feature_indices=fidx)

    fidx = f.list_bond_features(name="bond_length", origin="rdkit").index.to_list()[0]
    f.featurize_bonds(bond_indices="all", feature_indices=fidx)

    for mol in f.mol_vault.mol_objects:
        for atom in mol.GetAtoms():
            assert atom.GetPropsAsDict() != {}
        for bond in mol.GetBonds():
            assert bond.GetPropsAsDict() != {}

    # Clear the atom feature cache and check the data
    if feature_type.strip().lower() == "atom":
        f.clear_atom_feature_cache(origin=origin)

        for mol in f.mol_vault.mol_objects:
            for atom in mol.GetAtoms():
                assert atom.GetPropsAsDict() == {}
            for bond in mol.GetBonds():
                assert bond.GetPropsAsDict() != {}

        assert f.mol_vault.atom_feature_cache_n == [{} for _ in f.mol_vault.mol_objects]
        assert all(len(bond_feat_dict) > 0 for bond_feat_dict in f.mol_vault.bond_feature_cache)

    # Check bond feature data
    if feature_type.strip().lower() == "bond":
        f.clear_bond_feature_cache(origin=origin)

        for mol in f.mol_vault.mol_objects:
            for atom in mol.GetAtoms():
                assert atom.GetPropsAsDict() != {}
            for bond in mol.GetBonds():
                assert bond.GetPropsAsDict() == {}

        assert f.mol_vault.bond_feature_cache == [{} for _ in f.mol_vault.mol_objects]
        assert all(len(atom_feat_dict) > 0 for atom_feat_dict in f.mol_vault.atom_feature_cache_n)

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)

    # Clean up
    shutil.rmtree(path="irrelevant_out_dir")
    clean_up_logfile()


@pytest.mark.clear_feature_cache
@pytest.mark.parametrize(
    "feature_type, origin, _error_type",
    [
        ("atom", False, TypeError),
        ("atom", ["rdkit", None], TypeError),
        ("atom", ["qmdesc", "unfortunately_i_am_not_a_valid_origin"], ValueError),
        ("bond", False, TypeError),
        ("bond", ["rdkit", None], TypeError),
        ("bond", ["qmdesc", "unfortunately_i_am_not_a_valid_origin"], ValueError),
    ],
)
def test_clear_feature_cache2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    feature_type: Any,
    origin: Any,
    _error_type: Any,
) -> None:
    """Test for the ``clear_atom_feature_cache()`` and ``clear_bond_feature_cache()``
    method: invalid inputs.
    """
    # Setup featurizer
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Clear feature cache
    if feature_type == "atom":
        with pytest.raises(_error_type):
            f.clear_atom_feature_cache(origin=origin)
    if feature_type == "bond":
        with pytest.raises(_error_type):
            f.clear_bond_feature_cache(origin=origin)

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()
