"""Test functions for the ``bonafide.utils.io_`` module."""

import os
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

from bonafide.utils.io_ import (
    extract_energy_from_string,
    read_mol_object,
    read_sd_file,
    read_smarts,
    read_smiles,
    read_xyz_file,
    write_sd_file,
    write_xyz_file_from_coordinates_array,
)

#############################################
# Tests for the read_mol_object() function. #
#############################################


def _get_mol_obj() -> Chem.rdchem.Mol:
    """Generate a dummy RDKit mol object."""
    mol = Chem.MolFromSmiles("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O")
    return mol


def _get_mol_obj2() -> Chem.rdchem.Mol:
    """Generate a dummy RDKit mol object."""
    return "not a mol object"


def _get_mol_obj3() -> Chem.rdchem.Mol:
    """Generate a dummy RDKit mol object."""
    mol = Chem.MolFromSmiles("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=42)
    return mol


def _get_mol_obj4() -> Chem.rdchem.Mol:
    """Generate a dummy RDKit mol object."""
    mol = Chem.MolFromSmiles("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42)

    mol.GetConformer(0).SetIntProp("prop1", 10)
    mol.GetConformer(1).SetDoubleProp("prop2", 3.14)
    mol.GetConformer(1).SetProp("prop3", "test_value")
    mol.GetConformer(2).SetBoolProp("prop4", True)

    return mol


def _get_mol_obj5() -> Chem.rdchem.Mol:
    """Generate a dummy RDKit mol object."""
    mol = Chem.MolFromSmiles("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O")
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    return mol


@pytest.mark.read_mol_object
@pytest.mark.parametrize(
    "input_mol, n_conformers, expected_error",
    [
        (_get_mol_obj(), 0, None),
        (_get_mol_obj2(), 0, "the input value is not a valid RDKit mol object"),
        (_get_mol_obj3(), 5, None),
        (_get_mol_obj4(), 3, None),
        (_get_mol_obj5(), 1, "the conformer with index 0 is 2D; only 3D conformers are supported"),
    ],
)
def test_read_mol_object(input_mol, n_conformers, expected_error) -> None:
    """Test for the ``read_mol_object()`` function."""
    init_mol, processed_mols, error_message = read_mol_object(input_mol)

    assert init_mol == input_mol
    assert type(processed_mols) == list
    assert all(isinstance(mol, Chem.rdchem.Mol) for mol in processed_mols)
    assert init_mol not in processed_mols
    assert error_message == expected_error

    if n_conformers != 0 and expected_error is None:
        assert len(processed_mols) == n_conformers
        assert all(mol.GetNumConformers() == 1 for mol in processed_mols)
    if n_conformers == 0 and expected_error is None:
        assert len(processed_mols) == 1

    if n_conformers == 3:
        assert processed_mols[0].GetPropsAsDict() == {"prop1": 10}
        assert processed_mols[1].GetPropsAsDict() == {"prop2": 3.14, "prop3": "test_value"}
        assert processed_mols[2].GetPropsAsDict() == {"prop4": True}


#########################################
# Tests for the read_smiles() function. #
#########################################
# read_smiles() will never reach an empty string as input, as this is checked
# before calling the function.


@pytest.mark.read_smiles
def test_read_smiles() -> None:
    """Test for the ``read_smiles()`` function: valid SMILES without explicit hydrogen
    atoms.
    """
    smiles = "COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O"
    mol, error_message = read_smiles(smiles)
    assert mol is not None
    assert isinstance(mol, Chem.Mol)
    assert error_message is None
    assert mol.GetNumAtoms() == 21
    assert mol.GetNumBonds() == 23
    assert mol.GetAtomWithIdx(4).GetSymbol() == "N"
    assert mol.GetAtomWithIdx(12).GetSymbol() == "S"
    assert CalcExactMolWt(mol) == pytest.approx(321.05902743199994)


@pytest.mark.read_smiles
def test_read_smiles2() -> None:
    """Test for the ``read_smiles()`` function: valid SMILES with explicit hydrogen
    atoms.
    """
    smiles = (
        "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
        "c([H])c1Cl)C([H])([H])C2([H])[H]"
    )
    mol, error_message = read_smiles(smiles)
    assert mol is not None
    assert isinstance(mol, Chem.Mol)
    assert error_message is None
    assert mol.GetNumAtoms() == 37
    assert mol.GetNumBonds() == 39
    assert mol.GetAtomWithIdx(10).GetSymbol() == "N"
    assert mol.GetAtomWithIdx(2).GetSymbol() == "S"
    assert CalcExactMolWt(mol) == pytest.approx(321.05902743199994)


@pytest.mark.read_smiles
def test_read_smiles3() -> None:
    """Test for the ``read_smiles()`` function: parsable SMILES that results in a mol object that
    cannot be sanitized.
    """
    smiles = "COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)#O"
    mol, error_message = read_smiles(smiles)
    assert mol is None
    assert (
        error_message
        == "Generation of RDKit mol object from SMILES string failed as it resulted in None."
    )


@pytest.mark.read_smiles
def test_read_smiles4() -> None:
    """Test for the ``read_smiles()`` function: not parsable string."""
    smiles = "Gartenlaube"
    mol, error_message = read_smiles(smiles)
    assert mol is None
    assert (
        error_message
        == "Generation of RDKit mol object from SMILES string failed as it resulted in None."
    )


@pytest.mark.read_smiles
def test_read_smiles5() -> None:
    """Test for the ``read_smiles()`` function: wrong data type."""
    smiles = 42
    mol, error_message = read_smiles(smiles)
    assert mol is None
    assert error_message.startswith(
        "Generation of RDKit mol object from SMILES string failed: No registered converter was able "
        "to produce a C++ rvalue"
    )


#########################################
# Tests for the read_smarts() function. #
#########################################


@pytest.mark.read_smarts
@pytest.mark.parametrize(
    "input_smarts, _expected_error",
    [
        ("[c;H1]", None),
        ("cN  ", None),
        ("   ", "the SMARTS string must not be empty"),
        ("not a smarts", "generation of RDKit mol object failed for SMARTS string"),
    ],
)
def test_read_smarts(input_smarts: str, _expected_error: Optional[str]) -> None:
    """Test for the ``read_smarts()`` function."""
    smarts_mol, error_message = read_smarts(input_smarts)
    if _expected_error is None:
        assert smarts_mol is not None
        assert isinstance(smarts_mol, Chem.Mol)
        assert error_message is None
    else:
        assert smarts_mol is None
        assert error_message.startswith(_expected_error)


###########################################
# Tests for the read_xyz_file() function. #
###########################################
# read_xyz_file() will never reach an empty string are a non-existing file as input,
# as this is checked before.


@pytest.mark.read_xyz_file
def test_read_xyz_file(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_xyz_file()`` function: valid XYZ file with 7 conformers."""
    input_file = fetch_data_file(file_name="clopidogrel.xyz")
    xyz_blocks, error_message = read_xyz_file(input_file)
    assert error_message is None
    assert type(xyz_blocks) is list
    assert len(xyz_blocks) == 7

    assert "\nCl          -1.21145  -2.28768   1.50365\n" in xyz_blocks[0]
    assert "\nN            0.46730  -0.66053  -0.95916\n" in xyz_blocks[2]
    assert "\nH            1.15694  -1.97055   0.69106\n" in xyz_blocks[4]
    assert "\nS            5.00909  -0.49880  -0.71655\n" in xyz_blocks[-1]

    _line_count = xyz_blocks[0].count("\n")

    _last_atoms = [
        "\nH           -3.09680  -3.58362  -0.09334\n",
        "\nH           -4.24076   2.84904   1.56839\n",
        "\nH           -3.21554   3.09666  -1.50694\n",
        "\nH           -3.40638  -2.16943   2.31486\n",
        "\nH           -4.21122   2.92320   2.11969\n",
        "\nH            2.15322  -3.22554   2.88405\n",
        "\nH           -4.71474  -2.53884  -0.61198\n",
    ]

    for block_idx, block in enumerate(xyz_blocks):
        assert type(block) is str
        assert block.count("\n") == _line_count
        assert block.startswith("37\n")
        assert block.endswith(_last_atoms[block_idx])


@pytest.mark.read_xyz_file
def test_read_xyz_file2(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_xyz_file()`` function: valid XYZ file with 1 conformer."""
    input_file = fetch_data_file(file_name="Cr_III_cation-conf_00.xyz")
    xyz_blocks, error_message = read_xyz_file(input_file)
    assert error_message is None
    assert type(xyz_blocks) is list
    assert len(xyz_blocks) == 1

    assert "\nCr           0.49132  -0.38079  -0.29495\n" in xyz_blocks[0]
    assert "\nH           -0.17235  -2.76053  -0.95318\n" in xyz_blocks[-1]

    assert type(xyz_blocks[0]) is str
    assert xyz_blocks[0].count("\n") == 35
    assert xyz_blocks[0].startswith("33\n")
    assert xyz_blocks[0].endswith("\nH            2.74721  -2.79354  -0.16246\n")


@pytest.mark.read_xyz_file
def test_read_xyz_file3(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_xyz_file()`` function: generally valid XYZ file but it contains two
    different molecules.
    """
    input_file = fetch_data_file(file_name="mixture.xyz")
    _, error_message = read_xyz_file(input_file)
    assert (
        error_message == "the number of atoms specified in the first line of XYZ block with "
        "index 1 (6) does not match the number of atoms specified in the XYZ block with "
        "index 0 (37)"
    )


@pytest.mark.read_xyz_file
def test_read_xyz_file4(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_xyz_file()`` function: generally valid XYZ file but in which the
    atom orders are shuffled.
    """
    input_file = fetch_data_file(file_name="shuffled_atom_order.xyz")
    _, error_message = read_xyz_file(input_file)
    assert (
        error_message == "the elements in the XYZ block of conformer with index 1 are not "
        "identical and/or in the same order as found in the conformer with index 0"
    )


@pytest.mark.read_xyz_file
def test_read_xyz_file5(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_xyz_file()`` function: invalid XYZ file in which one block contains
    an invalid element.
    """
    input_file = fetch_data_file(file_name="invalid_element.xyz")
    _, error_message = read_xyz_file(input_file)
    assert (
        error_message
        == "element symbol '*' in xzy block with index 0 is not a valid element symbol"
    )


@pytest.mark.read_xyz_file
def test_read_xyz_file6(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_xyz_file()`` function: invalid XYZ file of a 2D molecule."""
    input_file = fetch_data_file(file_name="invalid_2d_mol.xyz")
    _, error_message = read_xyz_file(input_file)
    assert (
        error_message == "a line in xzy block with index 0 does not contain exactly one element "
        "symbol and three cartesian coordinates (x, y, z)"
    )


@pytest.mark.read_xyz_file
def test_read_xyz_file7(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_xyz_file()`` function: invalid XYZ file in which one block contains
    a non-numeric coordinate.
    """
    input_file = fetch_data_file(file_name="invalid_nan_contained.xyz")
    _, error_message = read_xyz_file(input_file)
    assert error_message.startswith(
        "one of the coordinates in xzy block with index 3 is not a valid float: could not convert "
        "string to float"
    )


@pytest.mark.read_xyz_file
def test_read_xyz_file8(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_xyz_file()`` function: invalid XYZ file in which one block has the
    number of atoms and doc line flipped.
    """
    input_file = fetch_data_file(file_name="invalid_natoms_doc_flipped.xyz")
    _, error_message = read_xyz_file(input_file)
    assert error_message.startswith(
        "the first line of XYZ block with index 5 is not a single valid integer specifying the "
        "number of atoms in the molecule:"
    )


@pytest.mark.read_xyz_file
def test_read_xyz_file9(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_xyz_file()`` functioninvalid XYZ file that is actually an SD file
    with the wrong extension.
    """
    input_file = fetch_data_file(file_name="invalid_sdf_mocking_xyz.xyz")
    _, error_message = read_xyz_file(input_file)
    assert error_message == (
        "the first line of the file is not a single valid integer defining the number of atoms "
        "in the molecule"
    )


@pytest.mark.read_xyz_file
def test_read_xyz_file10(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_xyz_file()`` function: invalid XYZ file because it is empty."""
    input_file = fetch_data_file(file_name="invalid_empty.xyz")
    _, error_message = read_xyz_file(input_file)
    assert error_message == "the file is empty or only contains empty lines"


@pytest.mark.read_xyz_file
def test_read_xyz_file11(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_xyz_file()`` function: invalid XYZ file that has a wrong number of
    atoms specified.
    """
    input_file = fetch_data_file(file_name="invalid_wrong_atom_number.xyz")
    _, error_message = read_xyz_file(input_file)
    assert (
        error_message == "the file contains fewer non-empty lines (40) than required for a single "
        "structure block with 40 atoms"
    )


@pytest.mark.read_xyz_file
def test_read_xyz_file12(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_xyz_file()`` function: invalid XYZ file that cannot be read."""
    input_file = fetch_data_file(file_name="invalid_not_readable.xyz")
    _, error_message = read_xyz_file(input_file)
    assert error_message.startswith("opening of the file failed")


##########################################
# Tests for the read_sd_file() function. #
##########################################
# read_sd_file() will never reach an empty string are a non-existing file as input,
# as this is checked before.


@pytest.mark.read_sd_file
def test_read_sd_file(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_sd_file()`` function: valid SD file with 5 conformers."""
    input_file = fetch_data_file(file_name="spiro_mol.sdf")
    sdf_mols, error_message, stereo_message = read_sd_file(input_file)
    assert type(sdf_mols) == list
    assert None not in sdf_mols
    assert len(sdf_mols) == 5
    assert error_message is None
    assert stereo_message is None

    for mol in sdf_mols:
        _mol = Chem.Mol(mol)
        _mol = Chem.AddHs(_mol)

        assert _mol.GetNumAtoms() == mol.GetNumAtoms()
        assert mol.GetNumAtoms() == 38
        assert mol.GetNumBonds() == 40
        assert mol.GetAtomWithIdx(7).GetSymbol() == "N"
        assert mol.GetAtomWithIdx(15).GetSymbol() == "F"
        assert CalcExactMolWt(mol) == pytest.approx(271.138385288)


@pytest.mark.read_sd_file
def test_read_sd_file2(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_sd_file()`` function: valid SD file with 1 conformer."""
    input_file = fetch_data_file(file_name="radical_cation-conf_00.sdf")
    sdf_mols, error_message, stereo_message = read_sd_file(input_file)
    assert type(sdf_mols) == list
    assert None not in sdf_mols
    assert len(sdf_mols) == 1
    assert error_message is None
    assert stereo_message is None

    mol = sdf_mols[0]
    _mol = Chem.Mol(mol)
    _mol = Chem.AddHs(_mol)

    assert _mol.GetNumAtoms() == mol.GetNumAtoms()
    assert mol.GetNumAtoms() == 17
    assert mol.GetNumBonds() == 17
    assert mol.GetAtomWithIdx(0).GetSymbol() == "S"
    assert mol.GetAtomWithIdx(2).GetSymbol() == "N"


@pytest.mark.read_sd_file
def test_read_sd_file3(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_sd_file()`` function: generally valid SD file but it contains two
    different molecules.
    """
    input_file = fetch_data_file(file_name="mixture.sdf")
    sdf_mols, error_message, stereo_message = read_sd_file(input_file)
    assert sdf_mols is None
    assert error_message.startswith(
        "validation of the file failed: the generated SMILES string of the conformer with index 1"
    )
    assert stereo_message is None


@pytest.mark.read_sd_file
def test_read_sd_file4(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_sd_file()`` function: generally valid SD file but in which the atom
    orders are shuffled.
    """
    input_file = fetch_data_file(file_name="shuffled_atom_order.sdf")
    _, error_message, stereo_message = read_sd_file(input_file)
    assert (
        error_message
        == "validation of the file failed: the elements of the conformer with index 1 are not "
        "identical and/or in the same order as found in the conformer with index 0"
    )
    assert stereo_message is None


@pytest.mark.read_sd_file
def test_read_sd_file5(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_sd_file()`` function: invalid SD file in which one block contains
    an invalid element.
    """
    input_file = fetch_data_file(file_name="invalid_element.sdf")
    _, error_message, stereo_message = read_sd_file(input_file)
    assert (
        error_message == "validation of the file failed: element symbol '*' in SDF block with "
        "index 1 is not a valid element symbol"
    )
    assert stereo_message is None


@pytest.mark.read_sd_file
def test_read_sd_file6(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_sd_file()`` function: invalid SD file of a 2D molecule."""
    input_file = fetch_data_file(file_name="invalid_2d_mol.sdf")
    _, error_message, stereo_message = read_sd_file(input_file)
    assert (
        error_message == "validation of the file failed: the conformer with index 0 is not 3D. "
        "Only SD files containing 3D information are supported"
    )
    assert stereo_message is None


@pytest.mark.read_sd_file
def test_read_sd_file7(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_sd_file()`` function: invalid SD file in which one block contains a
    non-numeric coordinate.
    """
    input_file = fetch_data_file(file_name="invalid_nan_contained.sdf")
    _, error_message, stereo_message = read_sd_file(input_file)
    assert (
        error_message == "validation of the file failed: generation of RDKit mol object from "
        "SDF block failed for conformer with index 1 as it resulted in None"
    )
    assert stereo_message is None


@pytest.mark.read_sd_file
def test_read_sd_file8(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_sd_file()`` function: invalid SD file in which one block is missing
    the header lines.
    """
    input_file = fetch_data_file(file_name="invalid_missing_header.sdf")
    _, error_message, stereo_message = read_sd_file(input_file)
    assert (
        error_message == "validation of the file failed: generation of RDKit mol object from "
        "SDF block failed for conformer with index 0 as it resulted in None"
    )
    assert stereo_message is None


@pytest.mark.read_sd_file
def test_read_sd_file9(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_sd_file()`` function: invalid SD file that is actually an XYZ file
    with the wrong extension.
    """
    input_file = fetch_data_file(file_name="invalid_xyz_mocking_sdf.sdf")
    _, error_message, stereo_message = read_sd_file(input_file)
    assert (
        error_message == "validation of the file failed: generation of RDKit mol object from SDF "
        "block failed for conformer with index 0 as it resulted in None"
    )
    assert stereo_message is None


@pytest.mark.read_sd_file
def test_read_sd_file10(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_sd_file()`` function: invalid SD file that is empty."""
    input_file = fetch_data_file(file_name="invalid_empty.sdf")
    _, error_message, stereo_message = read_sd_file(input_file)
    assert error_message.startswith("opening of the file failed: File error: Invalid input file")
    assert stereo_message is None


@pytest.mark.read_sd_file
def test_read_sd_file11(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_sd_file()`` function: SD file with conformers of mixed
    stereochemistry (E/Z mixture).
    """
    input_file = fetch_data_file(file_name="e_z_mixture.sdf")
    sdf_mols, error_message, stereo_message = read_sd_file(input_file)
    assert type(sdf_mols) == list
    assert None not in sdf_mols
    assert len(sdf_mols) == 3
    assert error_message is None
    assert (
        stereo_message
        == "File validation: the conformers with the following indices have different "
        "stereochemical information than the conformer with index 0 (but identical atom "
        "connectivity): [1, 2]. Ensure that this is intended."
    )


@pytest.mark.read_sd_file
def test_read_sd_file12(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``read_sd_file()`` function: SD file with conformers of mixed
    stereochemistry (R/S mixture).
    """
    input_file = fetch_data_file(file_name="r_s_mixture.sdf")
    sdf_mols, error_message, stereo_message = read_sd_file(input_file)
    assert type(sdf_mols) == list
    assert None not in sdf_mols
    assert len(sdf_mols) == 2
    assert error_message is None
    assert (
        stereo_message
        == "File validation: the conformers with the following indices have different "
        "stereochemical information than the conformer with index 0 (but identical atom "
        "connectivity): [1]. Ensure that this is intended."
    )


###########################################
# Tests for the write_sd_file() function. #
###########################################
# It is ensured before that this function operates on a valid RDKit mol object.


@pytest.mark.write_sd_file
def test_write_sd_file() -> None:
    """Test for the ``write_sd_file()`` function."""
    # Read SMILES string
    smiles = (
        "C1([H])([H])N(C([H])([H])C([H])([H])C2(C([H])(C2([H])[H])[H])[H])"
        "C([H])([H])C(F)(OC12C(C(=O)C(=C([H])C2([H])[H])[H])([H])[H])F"
    )
    mol, error_message = read_smiles(smiles)
    assert mol is not None
    assert error_message is None
    assert mol.GetNumAtoms() == 38

    # Generate one conformer
    AllChem.EmbedMultipleConfs(mol, 1)

    # Write SD file
    _sd_file_path = "_temp.sdf"
    write_sd_file(mol, _sd_file_path)

    # Read in written SD file
    sdf_mols, error_message, stereo_message = read_sd_file(_sd_file_path)
    assert error_message is None
    assert stereo_message is None
    assert type(sdf_mols) == list
    assert len(sdf_mols) == 1

    mol_read = sdf_mols[0]
    assert mol_read is not None

    # Remove temporary file
    os.remove(_sd_file_path)

    # Compare mol objects
    assert mol_read.GetNumAtoms() == mol.GetNumAtoms()
    assert mol_read.GetNumBonds() == mol.GetNumBonds()

    symbols_original = [atom.GetSymbol() for atom in mol.GetAtoms()]
    symbols_read = [atom.GetSymbol() for atom in mol_read.GetAtoms()]
    assert symbols_original == symbols_read


########################################################
# Tests for the extract_energy_from_string() function. #
########################################################


@pytest.mark.extract_energy_from_string
@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        # Normal energy in kcal/mol
        (
            "Energy = -123.456 kcal/mol\n",
            (-123.456, "kcal_mol", pytest.approx(-516.539904), None),
        ),
        # Normal energy in Eh
        (
            "Gibbs free energy: -61.24567 Eh (after optimization)\n",
            (-61.24567, "eh", pytest.approx(-160800.4845365588), None),
        ),
        # Normal energy in kJ/mol
        (
            "-1268465.3 KJ MOL-1\n",
            (-1268465.3, "kj_mol", pytest.approx(-1268465.3), None),
        ),
        # Integer energy in Eh
        (
            "The energy is 5 Hartree.\n",
            (5.0, "eh", pytest.approx(13127.4982), None),
        ),
        # Zero energy in kcal/mol
        (
            "0 kcal_mol-1\n",
            (0.0, "kcal_mol", pytest.approx(0.0), None),
        ),
        # Multiple numbers
        (
            "Energy after 15 optimization cycles: -47895.425kcalmol-1\n",
            (-47895.425, "kcal_mol", pytest.approx(-200394.45820000002), None),
        ),
        # More than one space
        (
            "Final   energy:    -2500.0    kJ/mol\n",
            (-2500.0, "kj_mol", pytest.approx(-2500.0), None),
        ),
        # With plus sign and extra space
        (
            "   +42.0   kcal-mol   ",
            (42.0, "kcal_mol", pytest.approx(175.728), None),
        ),
        # Multiple energies given (should extract first one)
        (
            "First energy: -447.13 eH, second energy: -78954.1 kcal/mol",
            (-447.13, "eh", pytest.approx(-1173939.6540332), None),
        ),
        # Negative zero
        (
            "Energy: -0.0 kcal/mol",
            (0.0, "kcal_mol", pytest.approx(0.0), None),
        ),
        # Scientific notation
        (
            "Energy: 1.23e2 kcal/mol",
            (123.0, "kcal_mol", pytest.approx(514.632), None),
        ),
        # Scientific notation
        (
            "-21.823E-3 EH\n",
            (-0.021823, "eh", pytest.approx(-57.296278643719994), None),
        ),
        # Value with comma as thousands separator
        (
            "E -> -6,234.56 kcal-mol-1",
            (-6234.56, "kcal_mol", pytest.approx(-26085.39904), None),
        ),
        # Value with comma as thousands separator
        (
            "+65,234,001.11   kcal_mol^-1",
            (65234001.11, "kcal_mol", pytest.approx(272939060.64424), None),
        ),
        # Value with comma as thousands separator
        (
            "5,120 KJ mol-1",
            (5120.0, "kj_mol", pytest.approx(5120.0), None),
        ),
        # Unit missing (should not match)
        (
            "-87956.145 converged\n",
            (
                None,
                None,
                None,
                "no valid energy value with a supported unit (kJ/mol, kcal/mol, Eh) could be "
                "extracted from the string '-87956.145 converged\n'",
            ),
        ),
        # Unit missing (should not match)
        (
            "Value: 1235 end.\n",
            (
                None,
                None,
                None,
                "no valid energy value with a supported unit (kJ/mol, kcal/mol, Eh) could be "
                "extracted from the string 'Value: 1235 end.\n'",
            ),
        ),
        # Unit at wrong place (should not match)
        (
            "Energy (Eh): -54125.785\n",
            (
                None,
                None,
                None,
                "no valid energy value with a supported unit (kJ/mol, kcal/mol, Eh) could be "
                "extracted from the string 'Energy (Eh): -54125.785\n'",
            ),
        ),
        # Example for electronvolt (should not match)
        (
            "energy: -13.6 eV\n",
            (
                None,
                None,
                None,
                "no valid energy value with a supported unit (kJ/mol, kcal/mol, Eh) could be "
                "extracted from the string 'energy: -13.6 eV\n'",
            ),
        ),
    ],
)
def test_extract_energy_from_string(
    input_string: str,
    expected_output: Tuple[
        Union[float, None], Union[str, None], Union[float, None], Union[str, None]
    ],
) -> None:
    """Test for the ``extract_energy_from_string()`` function."""
    energy_as_submitted, unit_as_submitted, energy, _errmsg = extract_energy_from_string(
        input_string
    )
    assert energy_as_submitted == expected_output[0]
    assert unit_as_submitted == expected_output[1]
    assert energy == expected_output[2]
    assert _errmsg == expected_output[3]


###################################################################
# Tests for the write_xyz_file_from_coordinates_array() function. #
###################################################################


@pytest.mark.write_xyz_file_from_coordinates_array
def test_write_xyz_file_from_coordinates_array() -> None:
    """Test for the ``write_xyz_file_from_coordinates_array()`` function."""
    # Dummy molecule
    _elements = ["C", "F", "H", "H", "H"]
    _coordinates = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.3524],
            [1.0267, 0.0000, -0.3630],
            [-0.5133, -0.8892, -0.3630],
            [-0.5133, 0.8892, -0.3630],
        ]
    )
    _xyz_file_path = "_temp.xyz"

    # Write file
    _ = write_xyz_file_from_coordinates_array(
        elements=_elements, coordinates=_coordinates, file_path=_xyz_file_path
    )

    # Read in written file
    xyz_blocks, error_message = read_xyz_file(_xyz_file_path)

    # Remove temporary file
    os.remove(_xyz_file_path)

    # Check data from read file
    assert error_message is None
    assert type(xyz_blocks) is list
    assert len(xyz_blocks) == 1

    block = xyz_blocks[0]
    assert type(block) is str
    assert block.count("\n") == 7
    assert block.startswith("5\n")
