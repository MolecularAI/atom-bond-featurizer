"""Test functions for the ``list_atom_features()`` and ``list_bond_features()`` method in the
``bonafide.bonafide`` module.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Callable

import pandas as pd
import pytest
from rdkit import RDLogger

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


@pytest.mark.list_features
@pytest.mark.parametrize("feature_type", ["atom", "bond"])
def test_list_features(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    feature_type: str,
) -> None:
    """Test for the ``list_atom_features()`` and ``list_bond_features()`` method: valid
    examples.
    """
    f = fresh_featurizer()

    if feature_type == "atom":
        m = f.list_atom_features
    else:
        m = f.list_bond_features

    res_df = m()
    assert type(res_df) == pd.core.frame.DataFrame
    assert len(f._feature_info_df.columns) == 11
    assert f._feature_info_df.index.name == "INDEX"

    res_df2 = m(origin="rdkit")
    assert type(res_df2) == pd.core.frame.DataFrame
    assert len(f._feature_info_df.columns) == 11
    assert f._feature_info_df.index.name == "INDEX"
    assert len(res_df2) < len(res_df)

    res_df3 = m(origin="kallisto")
    assert type(res_df3) == pd.core.frame.DataFrame
    assert len(f._feature_info_df.columns) == 11
    assert f._feature_info_df.index.name == "INDEX"
    assert len(res_df3) < len(res_df2)

    res_df4 = m(origin="i_dont_exist")
    assert type(res_df4) == pd.core.frame.DataFrame
    assert len(f._feature_info_df.columns) == 11
    assert f._feature_info_df.index.name == "INDEX"
    assert len(res_df4) == 0

    res_df5 = m(dimensionality=None)
    assert type(res_df5) == pd.core.frame.DataFrame
    assert len(f._feature_info_df.columns) == 11
    assert f._feature_info_df.index.name == "INDEX"
    assert len(res_df5) == 0

    res_df6 = m(name="proximity_shell")
    assert type(res_df6) == pd.core.frame.DataFrame
    assert len(f._feature_info_df.columns) == 11
    assert f._feature_info_df.index.name == "INDEX"

    if feature_type == "atom":
        assert len(res_df6) == 1
    else:
        assert len(res_df6) == 0

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.list_features
@pytest.mark.parametrize("feature_type", ["atom", "bond"])
def test_list_features2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    feature_type: str,
) -> None:
    """Test for the ``list_atom_features()`` and ``list_bond_features()`` method: invalid
    examples.
    """
    f = fresh_featurizer()

    if feature_type == "atom":
        m = f.list_atom_features
    else:
        m = f.list_bond_features

    with pytest.raises(ValueError, match="Invalid input to "):
        m(i_dont_exist="test")
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    caplog.clear()

    with pytest.raises(ValueError, match="Invalid input: "):
        m(feature_type="irrelevant")
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    caplog.clear()

    with pytest.raises(ValueError, match="Invalid input to "):
        m(INDEX=5)
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.list_features
@pytest.mark.parametrize("feature_type", ["atom", "bond"])
def test_list_features3(
    fresh_featurizer: AtomBondFeaturizer, clean_up_logfile: Callable[[str], None], feature_type: str
) -> None:
    """Test for the ``list_atom_features()`` and ``list_bond_features()`` method: check feature
    index lists for correct dimensionality.
    """
    f = fresh_featurizer()

    if feature_type == "atom":
        feature_df = f.list_atom_features()
        assert (
            f._atom_feature_indices_2D
            == feature_df[
                (feature_df["feature_type"] == "atom") & (feature_df["dimensionality"] == "2D")
            ].index.tolist()
        )
        assert (
            f._atom_feature_indices_3D
            == feature_df[feature_df["feature_type"] == "atom"].index.tolist()
        )

    else:
        feature_df = f.list_bond_features()
        assert (
            f._bond_feature_indices_2D
            == feature_df[
                (feature_df["feature_type"] == "bond") & (feature_df["dimensionality"] == "2D")
            ].index.tolist()
        )
        assert (
            f._bond_feature_indices_3D
            == feature_df[feature_df["feature_type"] == "bond"].index.tolist()
        )

    # Clean up
    clean_up_logfile()
