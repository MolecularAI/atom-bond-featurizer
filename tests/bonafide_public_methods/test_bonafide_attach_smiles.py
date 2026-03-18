"""Test functions for the ``attach_smiles()`` method in the ``bonafide.bonafide`` module."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Union

import pytest
from rdkit import Chem

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer


@pytest.mark.attach_smiles
@pytest.mark.parametrize(
    "input_string, smiles, align, connectivity_method, covalent_radius_factor",
    [
        (
            "clopidogrel-conf_01.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            True,
            "connect_the_dots",
            1.3,
        ),
        (
            "clopidogrel-conf_01.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            False,
            "connect_the_dots",
            1.3,
        ),
        (
            "clopidogrel-conf_01.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            True,
            "van_der_waals",
            1.3,
        ),
        (
            "clopidogrel-conf_01.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            False,
            "van_der_waals",
            1.3,
        ),
        (
            "clopidogrel-conf_01.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            True,
            "hueckel",
            1.3,
        ),
        (
            "clopidogrel-conf_01.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            False,
            "hueckel",
            1.3,
        ),
        (
            "clopidogrel.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            True,
            "connect_the_dots",
            1.3,
        ),
        (
            "clopidogrel.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            False,
            "connect_the_dots",
            1.3,
        ),
        (
            "clopidogrel.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            True,
            "van_der_waals",
            1.3,
        ),
        (
            "clopidogrel.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            False,
            "van_der_waals",
            1.3,
        ),
        (
            "clopidogrel.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            True,
            "hueckel",
            1.3,
        ),
        (
            "clopidogrel.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            False,
            "hueckel",
            1.3,
        ),
    ],
)
def test_attach_smiles(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    input_string: str,
    smiles: str,
    align: bool,
    connectivity_method: str,
    covalent_radius_factor: Union[int, float],
) -> None:
    """Test for the ``attach_smiles()`` method: valid examples."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name=input_string)
    f.read_input(
        input_value=input_string,
        namespace="irrelevant",
        input_format="file",
        output_directory="irrelevant_out_dir",
    )
    assert f.mol_vault is not None
    assert f.mol_vault.smiles is None
    assert f.mol_vault.bonds_determined is False

    # Set charge if required
    if connectivity_method == "hueckel":
        f.set_charge(charge=0)

    # Attach SMILES
    f.attach_smiles(
        smiles=smiles,
        align=align,
        connectivity_method=connectivity_method,
        covalent_radius_factor=covalent_radius_factor,
    )
    assert f.mol_vault.smiles == smiles
    assert f.mol_vault.bonds_determined is True

    _mols = f.mol_vault.mol_objects
    assert all(
        [
            len(mol.GetBonds()) == len(Chem.MolFromSmiles(smiles, sanitize=False).GetBonds())
            for mol in _mols
        ]
    )
    assert all([len(mol.GetConformers()) == 1] for mol in _mols)

    # Check atom order
    if align is True:
        all([mol.GetAtomWithIdx(1).GetSymbol() == "O" for mol in _mols])
        all([mol.GetAtomWithIdx(4).GetSymbol() == "N" for mol in _mols])
        all([mol.GetAtomWithIdx(19).GetSymbol() == "Cl" for mol in _mols])
    else:
        all([mol.GetAtomWithIdx(14).GetSymbol() == "O" for mol in _mols])
        all([mol.GetAtomWithIdx(10).GetSymbol() == "N" for mol in _mols])
        all([mol.GetAtomWithIdx(30).GetSymbol() == "Cl" for mol in _mols])

    # Check logs
    assert len(caplog.records) > 0

    if align is True:
        assert all(record.levelno == logging.INFO for record in caplog.records)
    else:
        assert all(record.levelno in (logging.INFO, logging.WARNING) for record in caplog.records)

    # Clean up
    os.rmdir(path="irrelevant_out_dir")
    clean_up_logfile()


@pytest.mark.attach_smiles
def test_attach_smiles2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``attach_smiles()`` method: fails because no molecule read yet."""
    f = fresh_featurizer()
    assert f.mol_vault is None

    with pytest.raises(ValueError, match="attaching a SMILES string"):
        f.attach_smiles(smiles="CCO")
    assert f.mol_vault is None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.attach_smiles
def test_attach_smiles3(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``attach_smiles()`` method: fails because molecule in vault is 2D."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    f.read_input(input_value="O(C([H])([H])F)[H]", namespace="irrelevant")
    assert f.mol_vault is not None

    with pytest.raises(
        ValueError, match="Attaching a SMILES string to a 2D ensemble is not allowed."
    ):
        f.attach_smiles(smiles="CCO")
    assert f.mol_vault is not None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.attach_smiles
def test_attach_smiles4(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
) -> None:
    """Test for the ``attach_smiles()`` method: fails because bonds were already determined."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name="clopidogrel-conf_01.xyz")
    f.read_input(input_value=input_string, namespace="irrelevant", input_format="file")
    assert f.mol_vault is not None

    # Determine bonds before attaching the SMILES
    f.determine_bonds()

    with pytest.raises(
        ValueError,
        match="A SMILES string can only be attached to a molecule vault that has its bonds not "
        "yet determined.",
    ):
        f.attach_smiles(smiles="CCO")

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.attach_smiles
@pytest.mark.parametrize(
    "input_string, smiles, align, connectivity_method, covalent_radius_factor, _error_type",
    [
        (
            "clopidogrel-conf_01.xyz",
            [
                "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
                "c([H])c1Cl)C([H])([H])C2([H])[H]"
            ],
            True,
            "connect_the_dots",
            1.3,
            TypeError,
        ),
        (
            "clopidogrel-conf_01.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            None,
            "connect_the_dots",
            1.3,
            TypeError,
        ),
        (
            "clopidogrel-conf_01.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            False,
            ["connect_the_dots", "van_der_waals"],
            1.3,
            TypeError,
        ),
        (
            "clopidogrel-conf_01.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            False,
            "connect_the_dots",
            [1.3, 1.5],
            TypeError,
        ),
        (
            "clopidogrel-conf_01.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            False,
            "van_der_waals",
            0,
            ValueError,
        ),
        (
            "clopidogrel-conf_01.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            False,
            "invalid_method",
            1.3,
            ValueError,
        ),
        (
            "clopidogrel-conf_01.xyz",
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            False,
            "hueckel",
            1.3,
            ValueError,
        ),
    ],
)
def test_attach_smiles5(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    input_string: Any,
    smiles: Any,
    align: Any,
    connectivity_method: Any,
    covalent_radius_factor: Any,
    _error_type: Any,
) -> None:
    """Test for the ``attach_smiles()`` method: invalid inputs."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name=input_string)
    f.read_input(input_value=input_string, namespace="irrelevant", input_format="file")
    assert f.mol_vault is not None

    # Attach SMILES
    with pytest.raises(_error_type):
        f.attach_smiles(
            smiles=smiles,
            align=align,
            connectivity_method=connectivity_method,
            covalent_radius_factor=covalent_radius_factor,
        )

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.attach_smiles
@pytest.mark.parametrize(
    "smiles",
    [
        "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
        "N1([C@](c2c([H])c(c([H])c(c2Cl)[H])[H])([H])C(OC([H])([H])[H])=S)C(C(c2sc([H])c([H])"
        "c2C1([H])[H])([H])[H])([H])[H]",
        "in_my_wildest_dreams_i_am_a_smiles",
    ],
)
def test_attach_smiles6(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    smiles: str,
) -> None:
    """Test for the ``attach_smiles()`` method: fails because molecule in the vault and SMILES
    don't match.
    """
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name="clopidogrel-conf_01.xyz")
    f.read_input(input_value=input_string, namespace="irrelevant", input_format="file")
    assert f.mol_vault is not None

    # Attach SMILES
    with pytest.raises(ValueError):
        f.attach_smiles(smiles=smiles)

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()
