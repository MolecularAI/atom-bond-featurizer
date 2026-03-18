"""Test functions for the ``read_input()`` method in the ``bonafide.bonafide`` module."""

from __future__ import annotations

import logging
import os
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Tuple, Union

import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from bonafide.utils.molecule_vault import MolVault

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def _get_mol_obj() -> Chem.rdchem.Mol:
    """Generate a dummy RDKit mol object."""
    mol = Chem.MolFromSmiles("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42)

    mol.GetConformer(0).SetProp("ENERGY", "-123456.789 kj/mol")
    mol.GetConformer(1).SetProp("energy  ", "-1123456.789 kj/mol")
    mol.GetConformer(2).SetProp("eNERGY", "-1123456.789 kj/mol")

    return mol


def _get_mol_obj2() -> Chem.rdchem.Mol:
    """Generate a dummy RDKit mol object."""
    mol = Chem.MolFromSmiles("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O")
    return mol


@pytest.mark.read_input
@pytest.mark.parametrize(
    "input_value, input_format, read_energy, prune_by_energy, output_directory",
    [
        ("COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1", "smiles", False, None, None),
        (
            "[H]C(N1C(=C(OC([H])([H])[H])[N+](C1=S)[H])[H])([H])[H]",
            "SMILES  ",
            False,
            None,
            "test_output_folder",
        ),
        ("spiro_mol_e.xyz", "file", False, None, None),
        ("spiro_mol_e.xyz", "  FILE ", True, None, "spiro_mol_e"),
        ("spiro_mol_e.xyz", "file", True, (3, "kcal/mol"), "None"),
        ("clopidogrel_e.sdf", "file", False, None, "clopidogrel_e_out_dir"),
        ("clopidogrel_e.sdf", "file", True, None, None),
        ("clopidogrel_e.sdf", "file   ", True, (1.54, "kJ_per_mol"), "clopidogrel_e"),
        (_get_mol_obj(), "mol_Object", False, None, None),
        (_get_mol_obj(), "mol_object", True, None, None),
        (_get_mol_obj2(), "mol_object", False, None, "test_dir"),
    ],
)
def test_read_input(
    caplog: pytest.LogCaptureFixture,
    fetch_data_file: Callable[[str], str],
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    input_value: str,
    input_format: str,
    read_energy: bool,
    prune_by_energy: Tuple[Union[int, float], str],
    output_directory: str,
) -> None:
    """Test for the ``read_input()`` method: valid examples."""
    f = fresh_featurizer()
    assert f._namespace is None
    assert f.mol_vault is None

    # Get file path if input class is file
    if input_format.lower().strip() == "file":
        input_value = fetch_data_file(file_name=input_value)

    # Read input
    _init_dir_contents = os.listdir()
    f.read_input(
        input_value=input_value,
        namespace="test_namespace",
        input_format=input_format,
        read_energy=read_energy,
        prune_by_energy=prune_by_energy,
        output_directory=output_directory,
    )
    assert f._namespace == "test_namespace"
    assert type(f.mol_vault) == MolVault
    assert f.mol_vault.namespace == "test_namespace"
    assert all([n.startswith("test_namespace__conf") for n in f.mol_vault.conformer_names])

    # Check output directory
    if output_directory is not None:
        assert os.listdir() != _init_dir_contents
        assert os.path.isdir(s=output_directory) is True
        assert f._keep_output_files is True
        assert f._output_directory == os.path.abspath(path=output_directory)
        os.rmdir(path=output_directory)
    else:
        assert f._keep_output_files is False
        assert f._output_directory == os.getcwd()
        assert os.listdir() == _init_dir_contents

    # Check logs
    assert len(caplog.records) > 0
    if output_directory is not None:
        assert all(record.levelno == logging.INFO for record in caplog.records)
    else:
        assert Counter([record.levelno == logging.WARNING for record in caplog.records])[True] == 1

    # Clean up
    clean_up_logfile()


@pytest.mark.read_input
def test_read_input2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``read_input()`` method: output directory already exists."""
    f = fresh_featurizer()
    assert f.mol_vault is None

    _dir_name = "i_already_exist"
    os.mkdir(path=_dir_name)

    # Read input
    _init_dir_contents = os.listdir()
    with pytest.raises(
        FileExistsError, match="already exists and can therefore not be used as output directory."
    ):
        f.read_input(input_value="CCO", namespace="irrelevant", output_directory=_dir_name)
    assert f.mol_vault is None
    assert f._output_directory is None
    assert os.listdir() == _init_dir_contents

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    os.rmdir(path=_dir_name)
    clean_up_logfile()


@pytest.mark.read_input
def test_read_input3(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``read_input()`` method: parent directory is not writable."""
    f = fresh_featurizer()
    assert f.mol_vault is None

    _parent_dir_name = "not_writable_dir"
    os.mkdir(path=_parent_dir_name)
    os.chmod(path=_parent_dir_name, mode=0o400)

    _dir_path = os.path.join(_parent_dir_name, "output_dir")

    # Read input
    _init_dir_contents = os.listdir(_parent_dir_name)
    with pytest.raises(
        PermissionError,
        match="is not writable and can therefore not be used as parent directory of the "
        "output directory.",
    ):
        f.read_input(input_value="CCO", namespace="irrelevant", output_directory=_dir_path)
    assert f.mol_vault is None
    assert f._output_directory is None
    assert os.listdir(_parent_dir_name) == _init_dir_contents

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    os.rmdir(path=_parent_dir_name)
    clean_up_logfile()


@pytest.mark.read_input
@pytest.mark.parametrize(
    "input_value, namespace, input_format, read_energy, prune_by_energy, "
    "output_directory, _error_type",
    [
        ("COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1", False, "smiles", False, None, None, TypeError),
        (
            ["COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1"],
            "irrelevant",
            "smiles",
            False,
            None,
            None,
            TypeError,
        ),
        (False, "irrelevant", "file", False, None, "out_dir", TypeError),
        ("COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1", "irrelevant", None, False, None, None, TypeError),
        (
            "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
            "irrelevant",
            "smiles",
            "True",
            None,
            "out_dir",
            TypeError,
        ),
        ("   ", "irrelevant", "smiles", False, None, None, ValueError),
        ("", "irrelevant", "smiles", False, None, None, ValueError),
        ("_irrelevant", "irrelevant", "xyz_file", False, None, None, ValueError),
        (
            "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
            "irrelevant",
            "smiles",
            False,
            None,
            False,
            TypeError,
        ),
        ("invalid_extension.mol", "irrelevant", "file", False, None, None, ValueError),
        ("no_extension", "irrelevant", "file", False, None, None, ValueError),
        ("i_dont_exist.sdf", "irrelevant", "file", False, None, None, FileNotFoundError),
        ("i am not an RDKit mol object", "irrelevant", "mol_object", False, None, None, TypeError),
    ],
)
def test_read_input4(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    input_value: Any,
    namespace: Any,
    input_format: Any,
    read_energy: Any,
    prune_by_energy: Any,
    output_directory: Any,
    _error_type: Any,
) -> None:
    """Test for the ``read_input()`` method: invalid inputs."""
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Read input
    with pytest.raises(_error_type):
        f.read_input(
            input_value=input_value,
            namespace=namespace,
            input_format=input_format,
            read_energy=read_energy,
            prune_by_energy=prune_by_energy,
            output_directory=output_directory,
        )
    assert f.mol_vault is None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.read_input
@pytest.mark.parametrize(
    "input_value, input_format",
    [
        ("mixture.xyz", "file"),
        ("invalid_empty.sdf", "file"),
        ("i-am-not-a-smiles", "smiles"),
        ("mixture.xyz", "smiles"),
    ],
)
def test_read_input5(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    input_value: str,
    input_format: str,
) -> None:
    """Test for the ``read_input()`` method: invalid files/SMILES."""
    f = fresh_featurizer()
    assert f.mol_vault is None

    if input_format == "file":
        input_value = fetch_data_file(file_name=input_value)

    with pytest.raises(ValueError):
        f.read_input(input_value=input_value, namespace="irrelevant", input_format=input_format)

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()
