"""Test functions for the ``bonafide.utils.molecule_vault`` module."""

from __future__ import annotations

import copy
import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import ipywidgets
import numpy as np
import pytest
from PIL import PngImagePlugin
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdPartialCharges

from bonafide.utils.constants import (
    EH_TO_KJ_MOL,
    KCAL_MOL_TO_KJ_MOL,
    UNDESIRED_ATOM_BOND_PROPERTIES,
)
from bonafide.utils.io_ import read_sd_file, read_xyz_file

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bonafide.utils.molecule_vault import MolVault

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


##########################################
# Tests for the initialize_mol() method. #
##########################################


@pytest.mark.initialize_mol
@pytest.mark.parametrize(
    "smiles, smiles_is_valid, charge",
    [
        ("C1(=S)[N+]([H])C(OC([H])([H])[H])=C(N1C([H])([H])[H])[H]", True, 1),
        ("COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1", True, 0),
        ("not_a_smiles", False, 0),
        (None, False, 0),
        ("COC(#O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1", False, 0),  # not sanitizable
    ],
)
def test_initialize_mol(
    caplog: pytest.LogCaptureFixture,
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    smiles: str,
    smiles_is_valid: bool,
    charge: int,
) -> None:
    """Test for the ``initialize_mol()`` method: with SMILES."""
    # Arrange mol vault
    mol_vault = fresh_mol_vault(mol_inputs=[smiles], namespace="test_smiles", input_type="smiles")

    # Invalid SMILES strings
    if smiles_is_valid is False:
        with pytest.raises(ValueError):
            mol_vault.initialize_mol()

        assert mol_vault.mol_objects == []
        assert mol_vault.conformer_names == []
        assert mol_vault.is_valid == [False]

        assert len(caplog.records) > 0
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Valid SMILES strings
    else:
        mol_vault.initialize_mol()

        # Check vault
        assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])
        assert len(mol_vault.mol_objects) == 1

        for mol in mol_vault.mol_objects:
            for atom in mol.GetAtoms():
                assert atom.GetPropsAsDict() == {}
            for bond in mol.GetBonds():
                assert bond.GetPropsAsDict() == {}

        assert mol_vault.conformer_names == ["test_smiles__conf-0"]
        assert mol_vault.dimensionality == "2D"
        assert mol_vault.size == 0
        assert mol_vault.charge == charge
        assert all(mol_vault.is_valid)
        assert mol_vault.energies_n == [(None, "kj_mol")]
        assert mol_vault.energies_n_minus1 == [(None, "kj_mol")]
        assert mol_vault.energies_n_plus1 == [(None, "kj_mol")]
        assert mol_vault.energies_n_read is False
        assert mol_vault.boltzmann_weights == (None, [None])
        assert mol_vault.electronic_strucs_n == [None]
        assert mol_vault.electronic_strucs_n_minus1 == [None]
        assert mol_vault.electronic_strucs_n_plus1 == [None]
        assert mol_vault.electronic_struc_types_n == [None]
        assert mol_vault.electronic_struc_types_n_minus1 == [None]
        assert mol_vault.electronic_struc_types_n_plus1 == [None]
        assert mol_vault.bonds_determined is True

        _expected_cache = [{}]
        assert mol_vault.atom_feature_cache_n == _expected_cache
        assert mol_vault.atom_feature_cache_n_minus1 == _expected_cache
        assert mol_vault.atom_feature_cache_n_plus1 == _expected_cache
        assert mol_vault.bond_feature_cache == _expected_cache
        assert mol_vault.global_feature_cache == _expected_cache

        # Check logs
        assert len(caplog.records) > 0
        assert all(record.levelno == logging.INFO for record in caplog.records)

    # Checks for both valid and invalid cases
    assert mol_vault.namespace == "test_smiles"
    assert mol_vault.input_type == "smiles"
    assert mol_vault.smiles == smiles


@pytest.mark.initialize_mol
@pytest.mark.parametrize(
    "input_file_name, expected_vault_size, bonds_included",
    [
        ("clopidogrel.xyz", 7, False),
        ("spiro_mol.sdf", 5, True),
        ("Cr_III_cation-conf_00.xyz", 1, False),
        ("radical_cation-conf_00.sdf", 1, True),
    ],
)
def test_initialize_mol2(
    caplog: pytest.LogCaptureFixture,
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    fetch_data_file: Callable[[str], str],
    input_file_name: str,
    expected_vault_size: int,
    bonds_included: bool,
) -> None:
    """Test for the ``initialize_mol()`` method: with files (all valid)."""
    # Preprocess input file
    input_file = fetch_data_file(file_name=input_file_name)

    _input_name = os.path.splitext(os.path.basename(input_file))[0]
    _file_type = os.path.splitext(input_file)[-1][1:]

    if _file_type == "xyz":
        mol_inputs, error_message = read_xyz_file(file_path=input_file)
    if _file_type == "sdf":
        mol_inputs, error_message = read_sd_file(file_path=input_file)

    assert error_message is None

    # Arrange mol vault
    mol_vault = fresh_mol_vault(mol_inputs=mol_inputs, namespace=_input_name, input_type=_file_type)
    mol_vault.initialize_mol()

    # Check vault
    assert mol_vault.namespace == _input_name

    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])
    assert len(mol_vault.mol_objects) == expected_vault_size

    for mol in mol_vault.mol_objects:
        for atom in mol.GetAtoms():
            assert atom.GetPropsAsDict() == {}
        for bond in mol.GetBonds():
            assert bond.GetPropsAsDict() == {}

    assert mol_vault.conformer_names == [
        f"{_input_name}__conf-{idx}" for idx in range(expected_vault_size)
    ]
    assert mol_vault.dimensionality == "3D"
    assert mol_vault.size == expected_vault_size
    assert mol_vault.elements is None
    assert mol_vault.charge is None
    assert mol_vault.multiplicity is None
    assert all(mol_vault.is_valid)
    assert mol_vault.energies_n == []
    assert mol_vault.energies_n_read is False
    assert mol_vault.boltzmann_weights == ()
    assert mol_vault.electronic_strucs_n == []
    assert mol_vault.electronic_strucs_n_minus1 == []
    assert mol_vault.electronic_strucs_n_plus1 == []
    assert mol_vault.electronic_struc_types_n == []
    assert mol_vault.electronic_struc_types_n_minus1 == []
    assert mol_vault.electronic_struc_types_n_plus1 == []
    assert mol_vault.smiles is None
    assert mol_vault.bonds_determined == bonds_included

    _expected_cache = [{}] * expected_vault_size
    assert mol_vault.atom_feature_cache_n == _expected_cache
    assert mol_vault.atom_feature_cache_n_minus1 == _expected_cache
    assert mol_vault.atom_feature_cache_n_plus1 == _expected_cache
    assert mol_vault.bond_feature_cache == _expected_cache
    assert mol_vault.global_feature_cache == _expected_cache

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)


def _get_mol_obj() -> Chem.rdchem.Mol:
    """Generate a dummy RDKit mol object."""
    mol = Chem.MolFromSmiles("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O")
    return mol, [Chem.Mol(mol)], 0


def _get_mol_obj2() -> Chem.rdchem.Mol:
    """Generate a dummy RDKit mol object."""
    mol = Chem.MolFromSmiles("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O")
    mol = Chem.AddHs(mol)
    n_conformers = 3
    AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, randomSeed=42)

    mols = []
    for conf in mol.GetConformers():
        mol_ = Chem.Mol(mol)
        mol_.RemoveAllConformers()
        mol_.AddConformer(conf=conf, assignId=True)
        mols.append(mol_)

    return mol, mols, n_conformers


@pytest.mark.initialize_mol
@pytest.mark.parametrize("inp_mol, processed_mols, n_conformers", [_get_mol_obj(), _get_mol_obj2()])
def test_initialize_mol3(
    caplog: pytest.LogCaptureFixture,
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    inp_mol: Chem.rdchem.Mol,
    processed_mols: List[Chem.rdchem.Mol],
    n_conformers: int,
):
    # Arrange mol vault
    mol_vault = fresh_mol_vault(
        mol_inputs=(inp_mol, processed_mols), namespace="test_mol", input_type="mol_object"
    )

    mol_vault.initialize_mol()

    # Check vault
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])
    assert all(mol_vault.is_valid)
    assert mol_vault.size == n_conformers

    assert mol_vault.mol_inputs == [inp_mol]
    assert mol_vault._input_mol_objects == inp_mol

    for mol in mol_vault.mol_objects:
        for atom in mol.GetAtoms():
            assert atom.GetPropsAsDict() == {}
        for bond in mol.GetBonds():
            assert bond.GetPropsAsDict() == {}

    # RDKit mol object from SMILES (no conformers)
    if n_conformers == 0:
        assert len(mol_vault.mol_objects) == 1
        assert mol_vault.conformer_names == ["test_mol__conf-0"]
        assert mol_vault.dimensionality == "2D"
        assert mol_vault.charge == 0
        assert mol_vault.multiplicity is None

        assert mol_vault.energies_n == [(None, "kj_mol")]
        assert mol_vault.energies_n_minus1 == [(None, "kj_mol")]
        assert mol_vault.energies_n_plus1 == [(None, "kj_mol")]
        assert mol_vault.energies_n_read is False
        assert mol_vault.boltzmann_weights == (None, [None])
        assert mol_vault.electronic_strucs_n == [None]
        assert mol_vault.electronic_strucs_n_minus1 == [None]
        assert mol_vault.electronic_strucs_n_plus1 == [None]
        assert mol_vault.electronic_struc_types_n == [None]
        assert mol_vault.electronic_struc_types_n_minus1 == [None]
        assert mol_vault.electronic_struc_types_n_plus1 == [None]
        assert mol_vault.bonds_determined is True

        _expected_cache = [{}]
        assert mol_vault.atom_feature_cache_n == _expected_cache
        assert mol_vault.atom_feature_cache_n_minus1 == _expected_cache
        assert mol_vault.atom_feature_cache_n_plus1 == _expected_cache
        assert mol_vault.bond_feature_cache == _expected_cache
        assert mol_vault.global_feature_cache == _expected_cache

    # RDKit mol object with 3D conformers
    else:
        assert len(mol_vault.mol_objects) == n_conformers
        assert mol_vault.conformer_names == [
            "test_mol__conf-0",
            "test_mol__conf-1",
            "test_mol__conf-2",
        ]
        assert mol_vault.dimensionality == "3D"

        assert mol_vault.elements is None
        assert mol_vault.charge is None
        assert mol_vault.multiplicity is None
        assert all(mol_vault.is_valid)
        assert mol_vault.energies_n == []
        assert mol_vault.energies_n_read is False
        assert mol_vault.boltzmann_weights == ()
        assert mol_vault.electronic_strucs_n == []
        assert mol_vault.electronic_strucs_n_minus1 == []
        assert mol_vault.electronic_strucs_n_plus1 == []
        assert mol_vault.electronic_struc_types_n == []
        assert mol_vault.electronic_struc_types_n_minus1 == []
        assert mol_vault.electronic_struc_types_n_plus1 == []
        assert mol_vault.smiles is not None

        _expected_cache = [{}] * n_conformers
        assert mol_vault.atom_feature_cache_n == _expected_cache
        assert mol_vault.atom_feature_cache_n_minus1 == _expected_cache
        assert mol_vault.atom_feature_cache_n_plus1 == _expected_cache
        assert mol_vault.bond_feature_cache == _expected_cache
        assert mol_vault.global_feature_cache == _expected_cache

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)


@pytest.mark.initialize_mol
@pytest.mark.parametrize(
    "broken_xyz_block",
    [
        (
            "60\n"
            "fluoromethanol\n"
            "O            0.84258  -0.69380  -0.04401\n"
            "C           -0.32457   0.09824  -0.00222\n"
            "F           -1.37720  -0.74138  -0.14083\n"
            "H            1.59065  -0.08484   0.05589\n"
            "H           -0.40257   0.60940   0.96014\n"
            "H           -0.32889   0.81237  -0.82897\n"
        ),
        (False),
    ],
)
def test_initialize_mol4(
    caplog: pytest.LogCaptureFixture,
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    broken_xyz_block: str,
) -> None:
    """Test for the ``initialize_mol()`` method: broken XYZ block."""
    mol_vault = fresh_mol_vault(
        mol_inputs=[broken_xyz_block], namespace="irrelevant", input_type="xyz"
    )
    with pytest.raises(ValueError):
        mol_vault.initialize_mol()

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)


@pytest.mark.initialize_mol
def test_initialize_mol5(
    caplog: pytest.LogCaptureFixture,
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
) -> None:
    """Test for the ``initialize_mol()`` method: broken mol object (not sanitizable)."""
    # Engineer not sanitizable mol object for injection in the SDF workflow
    mol = Chem.MolFromSmiles("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)#O", sanitize=False)

    # Arrange mol vault with broken mol object
    mol_vault = fresh_mol_vault(mol_inputs=[mol], namespace="irrelevant", input_type="sdf")

    # Assert failure
    with pytest.raises(ValueError):
        mol_vault.initialize_mol()

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)


########################################
# Tests for the get_elements() method. #
########################################


@pytest.mark.get_elements
@pytest.mark.parametrize(
    "smiles, expected_elements",
    [
        (
            "COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O",
            np.array(
                [
                    "C",
                    "O",
                    "C",
                    "C",
                    "N",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "S",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "Cl",
                    "O",
                ]
            ),
        ),
        (
            "c12[c:1]([s:2][c:3]([H:4])[c:5]1[H:6])[C:7]([H:8])([H:36])[C:9]([H:10])([H:11])"
            "[N:12]([C@@:16]([c:17]1[c:18]([H:27])[c:19]([H:20])[c:21]([H:26])[c:22]([H:25])"
            "[c:23]1[Cl:24])([C:28](=[O:29])[O:30][C:31]([H:32])([H:33])[H:34])[H:35])[C:13]"
            "2([H:14])[H:15]",
            np.array(
                [
                    "C",
                    "C",
                    "S",
                    "C",
                    "H",
                    "C",
                    "H",
                    "C",
                    "H",
                    "H",
                    "C",
                    "H",
                    "H",
                    "N",
                    "C",
                    "C",
                    "C",
                    "H",
                    "C",
                    "H",
                    "C",
                    "H",
                    "C",
                    "H",
                    "C",
                    "Cl",
                    "C",
                    "O",
                    "O",
                    "C",
                    "H",
                    "H",
                    "H",
                    "H",
                    "C",
                    "H",
                    "H",
                ]
            ),
        ),
        (
            "CP1(C)CCN->[Cr+]<-12(Cl)(Cl)<-NCCP->2",
            np.array(["C", "P", "C", "C", "C", "N", "Cr", "Cl", "Cl", "N", "C", "C", "P"]),
        ),
    ],
)
def test_get_elements(
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    smiles: str,
    expected_elements: NDArray[np.str_],
) -> None:
    """Test for the ``get_elements()`` method: with SMILES strings."""
    # Arrange mol vault
    mol_vault = fresh_mol_vault(mol_inputs=[smiles], namespace="irrelevant", input_type="smiles")
    mol_vault.initialize_mol()
    assert mol_vault.elements is None

    # Get elements
    mol_vault.get_elements()
    assert isinstance(mol_vault.elements, np.ndarray)
    assert np.array_equal(mol_vault.elements, expected_elements)


@pytest.mark.get_elements
@pytest.mark.parametrize(
    "input_file_name, expected_elements",
    [
        (
            "clopidogrel.xyz",
            np.array(
                [
                    "C",
                    "O",
                    "C",
                    "C",
                    "N",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "S",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "Cl",
                    "O",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                ]
            ),
        ),
        (
            "spiro_mol.sdf",
            np.array(
                [
                    "O",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "N",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "F",
                    "F",
                    "O",
                    "C",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                ]
            ),
        ),
        (
            "radical_cation-conf_00.sdf",
            np.array(
                [
                    "S",
                    "C",
                    "N",
                    "C",
                    "O",
                    "C",
                    "C",
                    "N",
                    "C",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                ]
            ),
        ),
    ],
)
def test_get_elements2(
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    fetch_data_file: Callable[[str], str],
    input_file_name: str,
    expected_elements: NDArray[np.str_],
) -> None:
    """Test for the ``get_elements()`` method: with files."""
    # Preprocess input file
    input_file = fetch_data_file(file_name=input_file_name)

    _input_name = os.path.splitext(os.path.basename(input_file))[0]
    _file_type = os.path.splitext(input_file)[-1][1:]

    if _file_type == "xyz":
        mol_inputs, error_message = read_xyz_file(file_path=input_file)
    if _file_type == "sdf":
        mol_inputs, error_message = read_sd_file(file_path=input_file)

    assert error_message is None

    # Arrange mol vault
    mol_vault = fresh_mol_vault(mol_inputs=mol_inputs, namespace=_input_name, input_type=_file_type)
    mol_vault.initialize_mol()
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])
    assert mol_vault.elements is None

    # Get elements
    mol_vault.get_elements()
    assert isinstance(mol_vault.elements, np.ndarray)
    assert np.array_equal(mol_vault.elements, expected_elements)


#############################################
# Tests for the read_mol_energies() method. #
#############################################


@pytest.mark.read_mol_energies
@pytest.mark.parametrize(
    "input_file_name, expected_energies, expected_unit, conversion_factor",
    [
        (
            "clopidogrel_e.xyz",
            [
                -61.266811591674,
                -61.270523020326,
                -61.268679432880,
                -61.266797716181,
                -61.272837688782,
                -61.273281213938,
                -61.271402885312,
            ],
            "eh",
            EH_TO_KJ_MOL,
        ),
        (
            "spiro_mol_e.sdf",
            [
                -60.170032727877,
                -60.170180567671,
                -60.165602531234,
                -60.165793670553,
                -60.164372396614,
            ],
            "eh",
            EH_TO_KJ_MOL,
        ),
        (
            "Cr_III_cation_e_kcal.xyz",
            [-28100.085192614333, -28100.15842617021],
            "kcal_mol",
            KCAL_MOL_TO_KJ_MOL,
        ),
        (
            "radical_cation_e.sdf",
            [-72242.23058015583, -72222.83673838593],
            "kj_mol",
            1,
        ),
    ],
)
def test_read_mol_energies(
    caplog: pytest.LogCaptureFixture,
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    fetch_data_file: Callable[[str], str],
    input_file_name: str,
    expected_energies: List[float],
    expected_unit: str,
    conversion_factor: float,
) -> None:
    """Test for the ``read_mol_energies()`` method: working cases."""
    # Preprocess input file
    input_file = fetch_data_file(file_name=input_file_name)

    _input_name = os.path.splitext(os.path.basename(input_file))[0]
    _file_type = os.path.splitext(input_file)[-1][1:]

    if _file_type == "xyz":
        mol_inputs, error_message = read_xyz_file(file_path=input_file)
    if _file_type == "sdf":
        mol_inputs, error_message = read_sd_file(file_path=input_file)

    assert error_message is None

    # Arrange mol vault
    mol_vault = fresh_mol_vault(mol_inputs=mol_inputs, namespace=_input_name, input_type=_file_type)
    mol_vault.initialize_mol()
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])
    assert mol_vault.energies_n_read is False
    assert mol_vault.energies_n == []
    assert mol_vault._input_energies_n == []

    # Get energies
    mol_vault.read_mol_energies()

    assert mol_vault.energies_n_read is True
    assert len(mol_vault.energies_n) == len(expected_energies)
    assert len(mol_vault.energies_n) == len(mol_vault._input_energies_n)
    assert len(mol_vault.energies_n) == mol_vault.size

    assert all(isinstance(e, tuple) for e in mol_vault.energies_n)
    assert all(isinstance(e, tuple) for e in mol_vault._input_energies_n)

    assert all(isinstance(e[0], float) or e[0] is None for e in mol_vault.energies_n)
    assert all(isinstance(e[0], float) or e[0] is None for e in mol_vault._input_energies_n)

    assert all(e[1] == "kj_mol" for e in mol_vault.energies_n)
    assert all(e[1] == expected_unit for e in mol_vault._input_energies_n)

    assert all(
        pytest.approx(input_energy) == expected_energy
        for expected_energy, (input_energy, _) in zip(
            expected_energies, mol_vault._input_energies_n
        )
    )

    assert all(
        pytest.approx(energy) == expected_energy * conversion_factor
        for expected_energy, (energy, _) in zip(expected_energies, mol_vault.energies_n)
    )

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)


@pytest.mark.read_mol_energies
@pytest.mark.parametrize(
    "input_file_name, expected_error",
    [
        (
            "clopidogrel.xyz",
            "Extraction of energy from XYZ block resulted in None for conformer with index 0: "
            "no valid energy value with a supported unit",
        ),
        (
            "clopidogrel_e_one_missing.xyz",
            "Extraction of energy from XYZ block resulted in None for conformer with index 3: "
            "no valid energy value with a supported unit",
        ),
        (
            "clopidogrel_empty_doc.xyz",
            "Extraction of energy from XYZ block resulted in None for conformer with index 0: "
            "no valid energy value with a supported unit",
        ),
        (
            "spiro_mol_e_no_unit.sdf",
            "Extraction of energy from SDF mol resulted in None for conformer with index 0: "
            "no valid energy value with a supported unit",
        ),
        (
            "spiro_mol.sdf",
            "Extraction of energy from SDF mol resulted in None for conformer with index 0: "
            "no property named 'energy' was found in the RDKit mol object",
        ),
    ],
)
def test_read_mol_energies2(
    caplog: pytest.LogCaptureFixture,
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    fetch_data_file: Callable[[str], str],
    input_file_name: str,
    expected_error: str,
) -> None:
    """Test for the ``read_mol_energies()`` method: fails."""
    # Preprocess input file
    input_file = fetch_data_file(file_name=input_file_name)

    _input_name = os.path.splitext(os.path.basename(input_file))[0]
    _file_type = os.path.splitext(input_file)[-1][1:]

    if _file_type == "xyz":
        mol_inputs, error_message = read_xyz_file(file_path=input_file)
    if _file_type == "sdf":
        mol_inputs, error_message = read_sd_file(file_path=input_file)

    assert error_message is None

    # Arrange mol vault
    mol_vault = fresh_mol_vault(mol_inputs=mol_inputs, namespace=_input_name, input_type=_file_type)
    mol_vault.initialize_mol()
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])
    assert mol_vault.energies_n_read is False
    assert mol_vault.energies_n == []
    assert mol_vault._input_energies_n == []

    # Get energies
    with pytest.raises(ValueError, match=expected_error):
        mol_vault.read_mol_energies()

    assert mol_vault.energies_n_read is False

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)


######################################
# Tests for the render_mol() method. #
######################################


@pytest.mark.render_mol
@pytest.mark.parametrize(
    "idx_type, image_size",
    [
        ("atom", (412, 361)),
        ("bond", (578, 123)),
        (None, (300, 300)),
    ],
)
def test_render_mol(
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    idx_type: Optional[str],
    image_size: Tuple[int, int],
) -> None:
    """Test for the ``render_mol()`` method: 2D rendering."""
    # Arrange mol vault
    smiles = "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1"
    mol_vault = fresh_mol_vault(mol_inputs=[smiles], namespace="irrelevant", input_type="smiles")
    mol_vault.initialize_mol()
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])
    assert mol_vault.dimensionality == "2D"

    # Render mol
    disp = mol_vault.render_mol(idx_type=idx_type, in_3D=False, image_size=image_size)

    assert isinstance(disp, PngImagePlugin.PngImageFile)
    assert disp.size == image_size
    assert disp.format == "PNG"
    assert disp.mode == "RGB"


@pytest.mark.render_mol
@pytest.mark.parametrize(
    "idx_type, image_size",
    [
        ("atom", (426, 573)),
        ("bond", (485, 123)),
        (None, (11, 12)),
    ],
)
def test_render_mol2(
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    fetch_data_file: Callable[[str], str],
    idx_type: Optional[str],
    image_size: Tuple[int, int],
) -> None:
    """Test for the ``render_mol()`` method: 3D rendering."""
    # Preprocess input file with 3D coordinates
    input_file = fetch_data_file(file_name="clopidogrel_e.xyz")
    mol_inputs, error_message = read_xyz_file(file_path=input_file)
    assert error_message is None

    # Arrange mol vault
    mol_vault = fresh_mol_vault(mol_inputs=mol_inputs, namespace="irrelevant", input_type="xyz")
    mol_vault.initialize_mol()
    mol_vault.read_mol_energies()
    mol_vault.update_boltzmann_weights(temperature=315.2, ignore_invalid=True)
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])
    assert mol_vault.dimensionality == "3D"

    # Render mol in 3D
    disp = mol_vault.render_mol(idx_type=idx_type, in_3D=True, image_size=image_size)

    assert isinstance(disp, ipywidgets.VBox)
    assert len(disp.children) == 3

    # Check slider (first child)
    slider = disp.children[0]
    assert isinstance(slider, ipywidgets.IntSlider)
    assert slider.value == 0
    assert slider.min == 0
    assert slider.max == mol_vault.size - 1
    assert slider.step == 1
    assert slider.description == "Conformer index:"
    assert slider.continuous_update is False
    assert slider.orientation == "horizontal"
    assert slider.readout is True
    assert slider.readout_format == "d"

    # Check print output (second child)
    print_output = disp.children[1]
    assert isinstance(print_output, ipywidgets.Output)

    # Check viewer output (third child)
    viewer_output = disp.children[2]
    assert isinstance(viewer_output, ipywidgets.Output)

    # Verify slider has observer attached
    assert len(slider._trait_notifiers["value"]["change"]) > 0


####################################################
# Tests for the prune_ensemble_by_energy() method. #
####################################################


@pytest.mark.prune_ensemble_by_energy
@pytest.mark.parametrize(
    "input_file_name, energy_cutoff, expected_pruning_pattern",
    [
        ("clopidogrel_e.xyz", (10, "kJ_mol"), [False, True, False, False, True, True, True]),
        (
            "clopidogrel_e.xyz",
            (1.1, "kcal_per_mole"),
            [False, False, False, False, True, True, False],
        ),
        (
            "clopidogrel_e.xyz",
            (0, "kcal_per_mole"),
            [False, False, False, False, False, True, False],
        ),
    ],
)
def test_prune_ensemble_by_energy(
    caplog: pytest.LogCaptureFixture,
    fetch_data_file: Callable[[str], str],
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    input_file_name: str,
    energy_cutoff: Tuple[Union[int, float], str],
    expected_pruning_pattern: List[bool],
) -> None:
    """Test for the ``prune_ensemble_by_energy()`` method: working cases."""
    # Preprocess input file
    input_file = fetch_data_file(file_name=input_file_name)

    _input_name = os.path.splitext(os.path.basename(input_file))[0]
    _file_type = os.path.splitext(input_file)[-1][1:]

    if _file_type == "xyz":
        mol_inputs, error_message = read_xyz_file(file_path=input_file)
    if _file_type == "sdf":
        mol_inputs, error_message = read_sd_file(file_path=input_file)

    assert error_message is None

    # Arrange mol vault
    mol_vault = fresh_mol_vault(mol_inputs=mol_inputs, namespace=_input_name, input_type=_file_type)
    mol_vault.initialize_mol()
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])

    # Read energies
    mol_vault.read_mol_energies()
    assert mol_vault.energies_n_read is True
    assert all(mol_vault.is_valid)
    assert all(isinstance(e[0], float) or e[0] is None for e in mol_vault.energies_n)
    assert all(e[1] == "kj_mol" for e in mol_vault.energies_n)

    # Prune ensemble by energy
    mol_vault.prune_ensemble_by_energy(energy_cutoff=energy_cutoff, _called_from="irrelevant")

    # Check pruning
    assert mol_vault.is_valid == expected_pruning_pattern

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)


@pytest.mark.prune_ensemble_by_energy
@pytest.mark.parametrize(
    "set_read_true, energy_cutoff, expected_error_message, error_type",
    [
        (
            False,
            (10, "kJ_mol"),
            "The molecule vault does not contain energy information on the individual conformers. "
            "Therefore, the conformer ensemble cannot be pruned.",
            ValueError,
        ),
        (
            True,
            [10, "kJ_mol"],
            "Invalid input to 'prune_by_energy': must be of type tuple but obtained list.",
            TypeError,
        ),
        (
            True,
            (10, "kJ", "mol"),
            "Invalid input to 'prune_by_energy': must be a 2-tuple but obtained a tuple of "
            "length 3.",
            ValueError,
        ),
        (
            True,
            ("10", "kJ_mol"),
            "Invalid input to 'prune_by_energy': the first entry of the 2-tuple must be of type "
            "int or float but obtained str.",
            TypeError,
        ),
        (
            True,
            (10.5, False),
            "Invalid input to 'prune_by_energy': the second entry of the 2-tuple must be of type "
            "str but obtained bool.",
            TypeError,
        ),
        (
            True,
            (10.5, "not_a_valid_unit"),
            "no valid energy value with a supported unit",
            ValueError,
        ),
    ],
)
def test_prune_ensemble_by_energy2(
    caplog: pytest.LogCaptureFixture,
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    set_read_true: bool,
    energy_cutoff: Any,
    expected_error_message: str,
    error_type: Any,
) -> None:
    """Test for the ``prune_ensemble_by_energy()`` method: fails."""
    # Arrange dummy mol vault
    smiles = "CCO"
    mol_vault = fresh_mol_vault(mol_inputs=[smiles], namespace="irrelevant", input_type="smiles")
    mol_vault.initialize_mol()
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])

    if set_read_true is True:
        mol_vault.energies_n_read = True

    with pytest.raises(
        error_type,
        match=expected_error_message,
    ):
        mol_vault.prune_ensemble_by_energy(energy_cutoff=energy_cutoff, _called_from=None)

    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)


##############################################
# Tests for the compare_conformers() method. #
##############################################


@pytest.mark.compare_conformers
def test_compare_conformers(caplog: pytest.LogCaptureFixture, fresh_mol_vault: MolVault) -> None:
    """Test for the ``test_compare_conformers()`` method: working case."""
    # Arrange dummy mol vault
    smiles = "CCO"
    mol_vault = fresh_mol_vault(mol_inputs=[smiles], namespace="irrelevant", input_type="smiles")
    mol_vault.initialize_mol()
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])

    # Inject artificial mol objects
    mol0 = Chem.MolFromSmiles("COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1")
    mol1 = Chem.MolFromSmiles("COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1")
    mol2 = Chem.MolFromSmiles("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O")

    mol_vault.mol_objects = [mol0, mol1, mol2]
    mol_vault.size = len(mol_vault.mol_objects)

    # Compare conformers
    mol_vault.compare_conformers()

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)


@pytest.mark.compare_conformers
def test_compare_conformers2(caplog: pytest.LogCaptureFixture, fresh_mol_vault: MolVault) -> None:
    """Test for the ``test_compare_conformers()`` method: fail."""
    # Arrange dummy mol vault
    smiles = "CCO"
    mol_vault = fresh_mol_vault(mol_inputs=[smiles], namespace="irrelevant", input_type="smiles")
    mol_vault.initialize_mol()
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])

    # Inject artificial mol objects
    mol0 = Chem.MolFromSmiles("COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1")
    mol1 = Chem.MolFromSmiles("COC([O])[C@H](c1ccccc1Cl)N1CCc2sccc2C1")
    mol2 = Chem.MolFromSmiles("COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1")

    mol_vault.mol_objects = [mol0, mol1, mol2]
    mol_vault.size = len(mol_vault.mol_objects)

    # Compare conformers
    mol_vault.compare_conformers()

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.WARNING for record in caplog.records)


############################################
# Tests for the clean_properties() method. #
############################################


@pytest.mark.clean_properties
def test_clean_properties(fresh_mol_vault: MolVault) -> None:
    """Test for the ``clean_properties()`` method."""
    # Arrange dummy mol vault
    smiles = "CCO"
    mol_vault = fresh_mol_vault(mol_inputs=[smiles], namespace="irrelevant", input_type="smiles")
    mol_vault.initialize_mol()
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])

    # Inject artificial mol objects
    mol0 = Chem.MolFromSmiles("COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1")
    mol1 = Chem.MolFromSmiles("COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1")

    # Calculate Gasteiger charges for getting undesired properties
    rdPartialCharges.ComputeGasteigerCharges(mol0)
    rdPartialCharges.ComputeGasteigerCharges(mol1)

    # Set properties that should not be removed
    _idx = 0
    _prop = "prop_to_keep"
    _value = "relevant"

    atom = mol0.GetAtomWithIdx(_idx)
    atom.SetProp(_prop, _value)
    assert mol0.GetAtomWithIdx(_idx).GetProp(_prop) == _value

    atom = mol1.GetAtomWithIdx(_idx)
    atom.SetProp(_prop, _value)
    assert mol1.GetAtomWithIdx(_idx).GetProp(_prop) == _value

    bond = mol0.GetBondWithIdx(_idx)
    bond.SetProp(_prop, _value)
    assert mol0.GetBondWithIdx(_idx).GetProp(_prop) == _value

    bond = mol1.GetBondWithIdx(_idx)
    bond.SetProp(_prop, _value)
    assert mol1.GetBondWithIdx(_idx).GetProp(_prop) == _value

    mol_vault.mol_objects = [mol0, mol1]
    mol_vault.size = len(mol_vault.mol_objects)

    # Check that properties are there
    for mol in mol_vault.mol_objects:
        for atom in mol.GetAtoms():
            assert atom.HasProp("_GasteigerCharge") == 1
            assert atom.HasProp("_GasteigerHCharge") == 1

    # Clear properties
    mol_vault.clean_properties()

    for mol in mol_vault.mol_objects:
        # Check that undesired properties are gone
        for atom in mol.GetAtoms():
            for prop in UNDESIRED_ATOM_BOND_PROPERTIES:
                assert atom.HasProp(prop) == 0
        for bond in mol.GetBonds():
            for prop in UNDESIRED_ATOM_BOND_PROPERTIES:
                assert bond.HasProp(prop) == 0

        # Check that desired properties are still there
        assert mol.GetAtomWithIdx(_idx).GetProp(_prop) == _value
        assert mol.GetBondWithIdx(_idx).GetProp(_prop) == _value


################################################
# Tests for the clear_feature_cache_() method. #
################################################


# Explicit testing for props of atoms/bonds not included, very tedious to set up.
# This is tested in test_clear_feature_cache in test_bonafide_clear_feature_cache.py
@pytest.mark.clear_feature_cache_
@pytest.mark.parametrize(
    "cache_size, to_be_cleared, init_cache, cleared_cache",
    [
        (
            1,
            ["rdkit"],
            [
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 6,
                        1: 8,
                        2: 6,
                        3: 6,
                    },
                }
            ],
            [
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                    },
                }
            ],
        ),
        (
            1,
            ["bonafide"],
            [
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 6,
                        1: 8,
                        2: 6,
                        3: 6,
                    },
                }
            ],
            [
                {
                    "rdkit2D-atom-atomic_number": {
                        0: 6,
                        1: 8,
                        2: 6,
                        3: 6,
                    },
                }
            ],
        ),
        (
            1,
            ["rdkit", "bonafide"],
            [
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 6,
                        1: 8,
                        2: 6,
                        3: 6,
                    },
                }
            ],
            [{}],
        ),
        (
            1,
            None,
            [
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 6,
                        1: 8,
                        2: 6,
                        3: 6,
                    },
                }
            ],
            [{}],
        ),
        (
            1,
            ["dbstep"],
            [
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 6,
                        1: 8,
                        2: 6,
                        3: 6,
                    },
                }
            ],
            [
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 6,
                        1: 8,
                        2: 6,
                        3: 6,
                    },
                }
            ],
        ),
        (
            3,
            ["rdkit"],
            [
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 6,
                        1: 8,
                        2: 6,
                        3: 6,
                    },
                },
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0b",
                        1: "1b",
                        2: "2b",
                        3: "3b",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 62,
                        1: 82,
                        2: 62,
                        3: 62,
                    },
                },
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0c",
                        1: "1c",
                        2: "2c",
                        3: "3c",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 63,
                        1: 83,
                        2: 63,
                        3: 63,
                    },
                },
            ],
            [
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                    },
                },
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0b",
                        1: "1b",
                        2: "2b",
                        3: "3b",
                    },
                },
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0c",
                        1: "1c",
                        2: "2c",
                        3: "3c",
                    },
                },
            ],
        ),
        (
            2,
            ["rdkit", "bonafide"],
            [
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 6,
                        1: 8,
                        2: 6,
                        3: 6,
                    },
                },
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0ff",
                        1: "1ff",
                        2: "2ff",
                        3: "3ff",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 600,
                        1: 800,
                        2: 600,
                        3: 600,
                    },
                },
            ],
            [{}, {}],
        ),
        (
            4,
            None,
            [
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 6,
                        1: 8,
                        2: 6,
                        3: 6,
                    },
                },
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0ff",
                        1: "1ff",
                        2: "2ff",
                        3: "3ff",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 600,
                        1: 800,
                        2: 600,
                        3: 600,
                    },
                },
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "0ffcc",
                        1: "1ffcc",
                        2: "2ffcc",
                        3: "3ffcc",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 6001,
                        1: 8001,
                        2: 6001,
                        3: 6001,
                    },
                },
                {
                    "bonafide2D-atom-is_symmetric_to": {
                        0: "d0ffcc",
                        1: "d1ffcc",
                        2: "d2ffcc",
                        3: "d3ffcc",
                    },
                    "rdkit2D-atom-atomic_number": {
                        0: 96001,
                        1: 98001,
                        2: 96001,
                        3: 96001,
                    },
                },
            ],
            [{}, {}, {}, {}],
        ),
    ],
)
def test_clear_feature_cache_(
    caplog: pytest.LogCaptureFixture,
    fetch_data_file: Callable[[str], str],
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    cache_size: int,
    to_be_cleared: Optional[List[str]],
    init_cache: List[Dict[str, Dict[Dict[int, Any]]]],
    cleared_cache: List[Dict[str, Dict[Dict[int, Any]]]],
) -> None:
    """Test for the ``clear_feature_cache_()`` method."""
    # Preprocess dummy input file (must be from file to allow for multiple conformers)
    input_file = fetch_data_file(file_name="clopidogrel_e.xyz")
    mol_inputs, error_message = read_xyz_file(file_path=input_file)
    assert error_message is None

    # Arrange dummy mol vault
    mol_vault = fresh_mol_vault(mol_inputs=mol_inputs, namespace="irrelevant", input_type="xyz")
    mol_vault.initialize_mol()
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])

    # Assert that caches are empty
    assert mol_vault.atom_feature_cache_n == [{}] * mol_vault.size
    assert mol_vault.atom_feature_cache_n_minus1 == [{}] * mol_vault.size
    assert mol_vault.atom_feature_cache_n_plus1 == [{}] * mol_vault.size
    assert mol_vault.bond_feature_cache == [{}] * mol_vault.size
    assert mol_vault.global_feature_cache == [{}] * mol_vault.size

    # Change to dummy vault size
    mol_vault.size = cache_size

    # Inject artificial feature caches
    mol_vault.atom_feature_cache_n = copy.deepcopy(init_cache)
    mol_vault.atom_feature_cache_n_minus1 = copy.deepcopy(init_cache)
    mol_vault.atom_feature_cache_n_plus1 = [{}] * cache_size

    # Prepare bond features and inject them
    init_cache_bond = [
        {name.replace("-atom-", "-bond-"): features for name, features in feature_dict.items()}
        for feature_dict in init_cache
    ]
    cleared_cache_bond = [
        {name.replace("-atom-", "-bond-"): features for name, features in feature_dict.items()}
        for feature_dict in cleared_cache
    ]
    mol_vault.bond_feature_cache = copy.deepcopy(init_cache_bond)

    # Make artificial global cache
    _artificial_global_cache = [
        {"some_global_feature": {0: "I", 1: "will be", 3: "removed"}}
    ] * cache_size
    mol_vault.global_feature_cache = copy.deepcopy(_artificial_global_cache)

    assert mol_vault.atom_feature_cache_n != [{}] * cache_size
    assert mol_vault.atom_feature_cache_n_minus1 != [{}] * cache_size
    assert mol_vault.bond_feature_cache != [{}] * cache_size
    assert mol_vault.global_feature_cache != [{}] * cache_size

    # Clear atom cache
    mol_vault.clear_feature_cache_(feature_type="atom", origins=to_be_cleared)

    # Assert that caches were correctly modified
    assert mol_vault.atom_feature_cache_n == cleared_cache
    assert mol_vault.atom_feature_cache_n_minus1 == cleared_cache
    assert mol_vault.atom_feature_cache_n_plus1 == [{}] * cache_size
    assert mol_vault.bond_feature_cache == init_cache_bond
    assert mol_vault.global_feature_cache == [{}] * cache_size

    # Bring back global features
    mol_vault.global_feature_cache = copy.deepcopy(_artificial_global_cache)

    # Clear bond cache
    mol_vault.clear_feature_cache_(feature_type="bond", origins=to_be_cleared)
    assert mol_vault.atom_feature_cache_n == cleared_cache
    assert mol_vault.atom_feature_cache_n_minus1 == cleared_cache
    assert mol_vault.atom_feature_cache_n_plus1 == [{}] * cache_size
    assert mol_vault.bond_feature_cache == cleared_cache_bond
    assert mol_vault.global_feature_cache == [{}] * cache_size

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)


####################################################
# Tests for the update_boltzmann_weights() method. #
####################################################


@pytest.mark.update_boltzmann_weights
@pytest.mark.parametrize(
    "temperature, energies, expected_weights",
    [
        (298.15, [(-6342.85421, "kj_mol")], [1.0]),
        (
            298.15,
            [
                (-6342.85421, "kj_mol"),
                (-6342.85421, "kj_mol"),
                (-6342.85421, "kj_mol"),
                (-6342.85421, "kj_mol"),
            ],
            [0.25, 0.25, 0.25, 0.25],
        ),
        (
            1298,
            [(-63742.85421, "kj_mol"), (-63742.85421, "kj_mol")],
            [0.5, 0.5],
        ),
        (
            298.15,
            [
                (-12548.1579, "kj_mol"),
                (-12547.95693, "kj_mol"),
                (-12546.72009, "kj_mol"),
                (-12538.76547, "kj_mol"),
                (-12533.96844, "kj_mol"),
                (-12532.50495, "kj_mol"),
                (-12519.92932, "kj_mol"),
            ],
            [
                0.39844865138315716,
                0.36742120847982895,
                0.2230893138189796,
                0.009013417996504037,
                0.001301625750331224,
                0.0007212646403128364,
                4.51793088623673e-06,
            ],
        ),
        (
            200.7,
            [
                (-12548.1579, "kj_mol"),
                (-12547.95693, "kj_mol"),
                (-12546.72009, "kj_mol"),
                (-12538.76547, "kj_mol"),
                (-12533.96844, "kj_mol"),
                (-12532.50495, "kj_mol"),
                (-12519.92932, "kj_mol"),
            ],
            [
                0.43235938806463614,
                0.38330229627083,
                0.18266032300039478,
                0.0015538063924944556,
                8.768699372472024e-05,
                3.647981740212381e-05,
                1.9460517995084264e-08,
            ],
        ),
        (
            298.15,
            [(-631.5, "kj_mol"), (-637.8, "kj_mol"), (-628.3, "kj_mol"), (-611.9, "kj_mol")],
            [0.071567982, 0.908722128, 0.01968353, 2.63601e-05],
        ),
        (
            0.1,
            [(-631.5, "kj_mol"), (-637.8, "kj_mol"), (-628.3, "kj_mol"), (-611.9, "kj_mol")],
            [0.0, 1.0, 0.0, 0.0],
        ),
        (
            1e9,
            [(-631.5, "kj_mol"), (-637.8, "kj_mol"), (-628.3, "kj_mol"), (-611.9, "kj_mol")],
            [0.25, 0.25, 0.25, 0.25],
        ),
    ],
)
def test_update_boltzmann_weights(
    caplog: pytest.LogCaptureFixture,
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    temperature: float,
    energies: List[Tuple[float, str]],
    expected_weights: List[float],
) -> None:
    """Test for the ``update_boltzmann_weights()`` method: check numerical values of weights."""
    # Arrange dummy mol vault
    smiles = "CCO"
    mol_vault = fresh_mol_vault(mol_inputs=[smiles], namespace="irrelevant", input_type="smiles")
    mol_vault.initialize_mol()
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])

    assert mol_vault.boltzmann_weights == (None, [None])
    assert mol_vault.energies_n_read is False
    assert mol_vault.energies_n == [(None, "kj_mol")]

    # Inject artificial energies
    mol_vault.is_valid = [True]
    mol_vault.energies_n = energies
    mol_vault.energies_n_read = True

    # Calculate Boltzmann weights
    mol_vault.update_boltzmann_weights(temperature=temperature, ignore_invalid=True)

    assert type(mol_vault.boltzmann_weights) == tuple
    assert len(mol_vault.boltzmann_weights) == 2
    assert mol_vault.boltzmann_weights[0] == temperature
    assert mol_vault.boltzmann_weights[1] == pytest.approx(expected_weights, rel=2e-6)
    assert len(mol_vault.boltzmann_weights[1]) == len(energies)
    assert sum(mol_vault.boltzmann_weights[1]) == pytest.approx(1.0)

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)


@pytest.mark.update_boltzmann_weights
@pytest.mark.parametrize(
    "energies, expected_weights, invalid_mask, ignore_invalid, warning_expected",
    [
        (
            [(-631.5, "kj_mol"), (-637.8, "kj_mol"), (-628.3, "kj_mol"), (-611.9, "kj_mol")],
            [0.071567982, 0.908722128, 0.01968353, 2.63601e-05],
            [True, True, True, True],
            True,
            False,
        ),
        (
            [(-631.5, "kj_mol"), (-637.8, "kj_mol"), (-628.3, "kj_mol"), (-611.9, "kj_mol")],
            [None, None, None, None],
            [True, False, True, True],
            False,
            True,
        ),
        (
            [(-631.5, "kj_mol"), (-637.8, "kj_mol"), (-628.3, "kj_mol"), (-611.9, "kj_mol")],
            [0.784067161, None, 0.215644049, 0.00028879],
            [True, False, True, True],
            True,
            True,
        ),
        (
            [(-631.5, "kj_mol"), (None, "kj_mol"), (-628.3, "kj_mol"), (None, "kj_mol")],
            [0.784293657, None, 0.215706343, None],
            [True, False, True, False],
            True,
            True,
        ),
        (
            [(None, "kj_mol"), (None, "kj_mol"), (None, "kj_mol"), (None, "kj_mol")],
            [None, None, None, None],
            [False, False, False, False],
            True,
            True,
        ),
    ],
)
def test_update_boltzmann_weights2(
    caplog: pytest.LogCaptureFixture,
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    energies: List[Tuple[float, str]],
    expected_weights: List[Union[float, None]],
    invalid_mask: List[bool],
    ignore_invalid: bool,
    warning_expected: bool,
) -> None:
    """Test for the ``update_boltzmann_weights()`` method: check ensembles with
    invalid conformers.
    """
    # Arrange dummy mol vault
    smiles = "CCO"
    mol_vault = fresh_mol_vault(mol_inputs=[smiles], namespace="irrelevant", input_type="smiles")
    mol_vault.initialize_mol()
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])

    # Set vault size to match energies
    mol_vault.size = len(energies)

    assert mol_vault.boltzmann_weights == (None, [None])
    assert mol_vault.energies_n_read is False
    assert mol_vault.energies_n == [(None, "kj_mol")]

    # Inject artificial energies
    mol_vault.is_valid = invalid_mask
    mol_vault.energies_n = energies
    mol_vault.energies_n_read = True

    # Calculate Boltzmann weights
    _T = 298.15
    mol_vault.update_boltzmann_weights(temperature=_T, ignore_invalid=ignore_invalid)

    assert type(mol_vault.boltzmann_weights) == tuple
    assert len(mol_vault.boltzmann_weights) == 2
    assert mol_vault.boltzmann_weights[0] == _T
    assert len(mol_vault.boltzmann_weights[1]) == len(energies)

    # No warning expected, all conformers valid
    if warning_expected is False:
        assert sum(mol_vault.boltzmann_weights[1]) == pytest.approx(1.0)
        assert mol_vault.boltzmann_weights[1] == pytest.approx(expected_weights, rel=2e-6)

        # Check logs
        assert len(caplog.records) > 0
        assert all(record.levelno == logging.INFO for record in caplog.records)
        return

    # Invalid conformers and ignore_invalid is False
    if ignore_invalid is False:
        assert mol_vault.boltzmann_weights[1] == expected_weights
        assert set(mol_vault.boltzmann_weights[1]) == {None}

        # Check logs
        assert len(caplog.records) > 0
        assert any(record.levelno == logging.WARNING for record in caplog.records)
        return

    # Invalid conformers and ignore_invalid is True
    for idx, (calc_weight, mask) in enumerate(zip(mol_vault.boltzmann_weights[1], invalid_mask)):
        if mask is True:
            assert type(calc_weight) == float
            assert calc_weight == pytest.approx(expected_weights[idx], rel=2e-6)
        else:
            assert calc_weight is None

    if any(invalid_mask) is True:
        _weights = np.array(mol_vault.boltzmann_weights[1])
        _weights = np.where(_weights == None, np.nan, _weights)
        assert np.nansum(_weights) == pytest.approx(1.0)
    else:
        assert set(mol_vault.boltzmann_weights[1]) == {None}

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.WARNING for record in caplog.records)
