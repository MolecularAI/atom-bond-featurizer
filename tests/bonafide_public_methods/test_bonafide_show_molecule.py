"""Test functions for the ``show_molecule()`` method in the ``bonafide.bonafide`` module."""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable

import ipywidgets
import pytest
from PIL import PngImagePlugin
from rdkit import RDLogger

from bonafide.utils.constants import FEATURE_TYPES

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


@pytest.mark.show_molecule
@pytest.mark.parametrize(
    "input_string, input_format, in_3D",
    [
        ("O=C1C=CCC2(C1)CN(CCC1CC1)CC(F)(F)O2", "smiles", False),
        ("C1(=S)[N+]([H])C(OC([H])([H])[H])=C(N1C([H])([H])[H])[H]", "SmileS", False),
        ("spiro_mol_e.xyz", "file", True),
        ("clopidogrel_e.sdf", "file", True),
        ("spiro_mol_e.xyz", "File", False),
        ("clopidogrel_e.sdf", "  file", False),
    ],
)
def test_show_molecule(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    input_string: str,
    input_format: str,
    in_3D: bool,
) -> None:
    """Test for the ``show_molecule()`` method: valid examples."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    if input_format.lower().strip() == "file":
        input_string = fetch_data_file(file_name=input_string)

    f.read_input(
        input_value=input_string,
        namespace="irrelevant",
        input_format=input_format,
        output_directory="irrelevant_out_dir",
    )

    for idx_type in ["atom", "bond", None]:
        for size in [(400, 400), (300, 600), (50, 50)]:
            # Render 2D molecule
            if in_3D is False:
                disp = f.show_molecule(index_type=idx_type, image_size=size)
                assert isinstance(disp, PngImagePlugin.PngImageFile)
                assert disp.size == size
                assert disp.format == "PNG"
                assert disp.mode == "RGB"

            # Render 3D molecule
            else:
                disp = f.show_molecule(in_3D=in_3D)
                assert isinstance(disp, ipywidgets.VBox)

    # Check logs
    assert len(caplog.records) > 0
    assert all(
        record.levelno == logging.INFO or record.levelno == logging.DEBUG
        for record in caplog.records
    )

    # Clean up
    os.rmdir(path="irrelevant_out_dir")
    clean_up_logfile()


@pytest.mark.show_molecule
def test_show_molecule2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
):
    """Test for the ``show_molecule()`` method: fails because no molecule read yet."""
    f = fresh_featurizer()
    assert f.mol_vault is None

    with pytest.raises(ValueError, match="showing the molecule"):
        f.show_molecule()

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.show_molecule
@pytest.mark.parametrize(
    "input_string, input_format, index_type, in_3D, size, _error_type",
    [
        (
            "O=C1C=CCC2(C1)CN(CCC1CC1)CC(F)(F)O2",
            "smiles",
            FEATURE_TYPES,
            False,
            (400, 400),
            TypeError,
        ),
        ("spiro_mol_e.xyz", "file", "atom", "False", (400, 400), TypeError),
        ("O=C1C=CCC2(C1)CN(CCC1CC1)CC(F)(F)O2", "smiles", "atom", False, [400, 400], TypeError),
        ("O=C1C=CCC2(C1)CN(CCC1CC1)CC(F)(F)O2", "smiles", "a", False, (400, 400), ValueError),
        ("O=C1C=CCC2(C1)CN(CCC1CC1)CC(F)(F)O2", "smiles", "bond", True, (400, 400), ValueError),
        ("O=C1C=CCC2(C1)CN(CCC1CC1)CC(F)(F)O2", "smiles", "atom", False, (400,), ValueError),
        ("O=C1C=CCC2(C1)CN(CCC1CC1)CC(F)(F)O2", "smiles", "atom", False, (-400, 400), ValueError),
    ],
)
def test_show_molecule3(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    input_string: str,
    input_format: str,
    index_type: Any,
    in_3D: Any,
    size: Any,
    _error_type: Any,
) -> None:
    """Test for the ``show_molecule()`` method: invalid inputs."""
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

    # Render molecule
    with pytest.raises(_error_type):
        f.show_molecule(index_type=index_type, in_3D=in_3D, image_size=size)

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    os.rmdir(path="irrelevant_out_dir")
    clean_up_logfile()
