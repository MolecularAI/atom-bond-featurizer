"""Test functions for the ``add_custom_featurizer()`` method in the ``bonafide.bonafide``
module.
"""

from __future__ import annotations

import copy
import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
import pytest
from rdkit import RDLogger

from bonafide.utils.base_featurizer import BaseFeaturizer

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


class Custom2DAtomDummyExample(BaseFeaturizer):
    """Dummy custom feature factory class."""

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        self.results[self.atom_bond_idx] = {self.feature_name: float(self.dummy_option * 2)}


dummy_feature_config = {"dummy_option": 1.5}

dummy_feature_info_dict = {
    "name": "custom2D-atom-dummy_example",
    "origin": "custom",
    "feature_type": "atom",
    "dimensionality": "2D",
    "data_type": "float",
    "requires_electronic_structure_data": False,
    "requires_bond_data": False,
    "requires_charge": False,
    "requires_multiplicity": False,
    "config_path": dummy_feature_config,
    "factory": Custom2DAtomDummyExample,
}


@pytest.mark.add_custom_featurizer
def test_add_custom_featurizer(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``add_custom_featurizer()`` method: valid example."""
    # Setup featurizer
    f = fresh_featurizer()
    assert f.mol_vault is None

    feature_df = pd.concat([f.list_atom_features(), f.list_bond_features()])
    feature_df = feature_df.sort_index()
    _init_len = len(feature_df)
    assert dummy_feature_info_dict["origin"] not in feature_df["origin"].values
    assert str(dummy_feature_info_dict["factory"]) not in feature_df["factory"].values

    # Add featurizer
    f.add_custom_featurizer(custom_metadata=dummy_feature_info_dict)

    feature_df_new = pd.concat([f.list_atom_features(), f.list_bond_features()])
    feature_df_new = feature_df_new.sort_index()
    _custom_feature_idx = int(feature_df_new.index[-1])
    assert len(feature_df_new) == _init_len + 1
    assert dummy_feature_info_dict["origin"] in feature_df_new["origin"].values
    assert str(dummy_feature_info_dict["factory"]) in feature_df_new["factory"].values

    # Use featurizer
    f.read_input(input_value="CCO", namespace="irrelevant", output_directory="irrelevant_out_dir")
    f.featurize_atoms(atom_indices=1, feature_indices=_custom_feature_idx)

    assert f.mol_vault.atom_feature_cache_n[0][dummy_feature_info_dict["name"]][1] == 3.0
    assert (
        f.mol_vault.mol_objects[0]
        .GetAtomWithIdx(1)
        .GetPropsAsDict()[dummy_feature_info_dict["name"]]
        == 3.0
    )

    # Check logs
    assert len(caplog.records) > 0

    for record in caplog.records:
        if record.levelno == logging.WARNING:
            assert (
                "No configuration settings validation class implemented for 'custom'." in record.msg
            )
        if record.levelno == logging.ERROR:
            raise AssertionError("No error should be raised during this test.")

    # Clean up
    os.rmdir(path="irrelevant_out_dir")
    clean_up_logfile()


dummy_feature_info_dict_wrong = copy.deepcopy(dummy_feature_info_dict)
del dummy_feature_info_dict_wrong["requires_electronic_structure_data"]

dummy_feature_info_dict_wrong2 = [copy.deepcopy(dummy_feature_info_dict)]


@pytest.mark.add_custom_featurizer
@pytest.mark.parametrize(
    "custom_metadata, _error_type",
    [(dummy_feature_info_dict_wrong, KeyError), (dummy_feature_info_dict_wrong2, TypeError)],
)
def test_add_custom_featurizer2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    custom_metadata: Any,
    _error_type: Any,
) -> None:
    """Test for the ``add_custom_featurizer()`` method: invalid example."""
    # Setup featurizer
    f = fresh_featurizer()
    assert f.mol_vault is None

    feature_df = pd.concat([f.list_atom_features(), f.list_bond_features()])
    feature_df = feature_df.sort_index()
    _init_len = len(feature_df)
    assert dummy_feature_info_dict["origin"] not in feature_df["origin"].values
    assert str(dummy_feature_info_dict["factory"]) not in feature_df["factory"].values

    # Add featurizer
    with pytest.raises(_error_type):
        f.add_custom_featurizer(custom_metadata=custom_metadata)

    feature_df_new = pd.concat([f.list_atom_features(), f.list_bond_features()])
    feature_df_new = feature_df_new.sort_index()
    assert len(feature_df_new) == _init_len
    assert dummy_feature_info_dict["origin"] not in feature_df["origin"].values
    assert str(dummy_feature_info_dict["factory"]) not in feature_df["factory"].values

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()
