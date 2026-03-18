"""Test functions for the ``determine_bonds()`` method in the ``bonafide.bonafide`` module."""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Union

import pytest
from rdkit import Chem, RDLogger

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


@pytest.mark.determine_bonds
@pytest.mark.parametrize(
    "connectivity_method, covalent_radius_factor, allow_charged_fragments, embed_chiral",
    [
        ("_use_default", None, None, None),
        ("connect_the_dots", 1.3, True, True),
        ("van_der_waals", 1.3, True, True),
        ("hueckel", 1.3, True, True),
        ("connect_the_dots", 1.3, True, False),
        ("van_der_waals", 1.3, True, False),
        ("hueckel", 1.3, True, False),
        ("connect_the_dots", 1.3, False, False),
        ("van_der_waals", 1.3, False, False),
        ("hueckel", 1.3, False, False),
        ("connect_the_dots", 1, True, True),
    ],
)
def test_determine_bonds(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    connectivity_method: str,
    covalent_radius_factor: Union[int, float],
    allow_charged_fragments: bool,
    embed_chiral: bool,
) -> None:
    """Test for the ``determine_bonds()`` method."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name="spiro_mol_e.xyz")
    f.read_input(
        input_value=input_string,
        namespace="irrelevant",
        input_format="file",
        output_directory="irrelevant_out_dir",
    )
    assert f.mol_vault is not None
    assert f.mol_vault.bonds_determined is False

    _mols = f.mol_vault.mol_objects
    assert all([len(mol.GetBonds()) == 0 for mol in _mols])

    # Set charge if required
    if connectivity_method == "hueckel":
        f.set_charge(charge=0)

    # Determine bonds
    if connectivity_method == "_use_default":
        f.determine_bonds()
    else:
        f.determine_bonds(
            connectivity_method=connectivity_method,
            covalent_radius_factor=covalent_radius_factor,
            allow_charged_fragments=allow_charged_fragments,
            embed_chiral=embed_chiral,
        )

    _mols = f.mol_vault.mol_objects
    assert f.mol_vault.bonds_determined is True
    assert all([len(mol.GetBonds()) > 0 for mol in _mols])
    assert len(set([len(mol.GetBonds()) for mol in _mols])) == 1

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)

    # Clean up
    os.rmdir(path="irrelevant_out_dir")
    clean_up_logfile()


@pytest.mark.determine_bonds
def test_determine_bonds2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``determine_bonds()`` method: fails because no molecule read yet."""
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Determine bonds
    with pytest.raises(ValueError, match="determining bonds"):
        f.determine_bonds()
    assert f.mol_vault is None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.determine_bonds
def test_determine_bonds3(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``determine_bonds()`` method: fails because molecule in vault is 2D (which
    means that the bonds are already determined).
    """
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    f.read_input(input_value="O(C([H])([H])F)[H]", namespace="irrelevant")
    assert f.mol_vault is not None

    # Determine bonds
    with pytest.raises(ValueError, match="All bonds are already defined."):
        f.determine_bonds()
    assert f.mol_vault is not None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.determine_bonds
def test_determine_bonds4(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
) -> None:
    """Test for the ``determine_bonds()`` method: fails because bonds were already determined."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name="clopidogrel-conf_01.xyz")
    f.read_input(input_value=input_string, namespace="irrelevant", input_format="file")
    assert f.mol_vault is not None

    # Determine bonds before actually testing the method
    f.determine_bonds()

    with pytest.raises(
        ValueError,
        match="The bonds of the molecule in the molecule vault are already determined and cannot "
        "be determined again.",
    ):
        f.determine_bonds()

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.determine_bonds
@pytest.mark.parametrize(
    "connectivity_method, covalent_radius_factor, allow_charged_fragments, "
    "embed_chiral, _error_type",
    [
        (False, 1.3, True, True, TypeError),
        ("connect_the_dots", "1.3", True, True, TypeError),
        ("connect_the_dots", -1.3, True, True, ValueError),
        ("van_der_waals", 1.3, (True,), True, TypeError),
        ("van_der_waals", 1.3, True, None, TypeError),
        ("using_me_as_a_method_would_be_insane", 1.3, True, False, ValueError),
        ("hueckel", 1.3, True, False, ValueError),
    ],
)
def test_determine_bonds5(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    connectivity_method: Any,
    covalent_radius_factor: Any,
    allow_charged_fragments: Any,
    embed_chiral: Any,
    _error_type: Any,
) -> None:
    """Test for the ``determine_bonds()`` method: invalid inputs."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name="spiro_mol_e.xyz")
    f.read_input(input_value=input_string, namespace="irrelevant", input_format="file")
    assert f.mol_vault is not None
    assert f.mol_vault.bonds_determined is False

    # Determine bonds
    with pytest.raises(_error_type):
        f.determine_bonds(
            connectivity_method=connectivity_method,
            covalent_radius_factor=covalent_radius_factor,
            allow_charged_fragments=allow_charged_fragments,
            embed_chiral=embed_chiral,
        )
    assert f.mol_vault.bonds_determined is False

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.determine_bonds
def test_determine_bonds6(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
) -> None:
    """Test for the ``determine_bonds()`` method: bond determining fails."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name="clopidogrel-conf_01.xyz")
    f.read_input(input_value=input_string, namespace="irrelevant", input_format="file")
    assert f.mol_vault is not None
    assert f.mol_vault.bonds_determined is False

    # Modify molecule vault to contain a broken mol object
    f.mol_vault.mol_objects = [
        Chem.MolFromSmiles("COC(#O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1", sanitize=False)
    ]

    # Determine bonds
    f.determine_bonds()
    assert f.mol_vault.bonds_determined is True

    # Check logs
    assert len(caplog.records) > 0

    _log_found = False
    _target_str = (
        "Determining the chemical bonds failed for conformer with index 0 which is "
        "therefore set to be invalid:"
    )
    for record in caplog.records:
        if record.levelno == logging.WARNING and _target_str in record.msg:
            _log_found = True
            break
    assert _log_found is True

    # Clean up
    clean_up_logfile()


@pytest.mark.determine_bonds
def test_determine_bonds7(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
) -> None:
    """Test for the ``determine_bonds()`` method: conformers are not identical after
    determining the bonds.
    """
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name="spiro_mol_e_atom_moved.xyz")
    f.read_input(input_value=input_string, namespace="irrelevant", input_format="file")
    f.set_charge(charge=0)
    assert f.mol_vault is not None
    assert f.mol_vault.bonds_determined is False

    # Determine bonds
    f.determine_bonds(
        connectivity_method="van_der_waals",
        covalent_radius_factor=1.3,
        allow_charged_fragments=True,
        embed_chiral=True,
    )
    assert f.mol_vault.bonds_determined is True

    # Check logs
    assert len(caplog.records) > 0

    _log_found = False
    _target_str = (
        "The conformer with index 1 does not match the conformer with index 0. Ensure "
        "that all conformers represent the same molecule and that this mismatch will "
        "not break downstream tasks."
    )
    for record in caplog.records:
        if record.levelno == logging.WARNING and _target_str in record.msg:
            _log_found = True
            break
    assert _log_found is True

    # Clean up
    clean_up_logfile()
