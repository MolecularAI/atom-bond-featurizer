"""Test functions for the ``bonafide.utils.helper_functions_chemistry`` module."""

from typing import Callable, Dict, List, Tuple

import pytest
from mendeleev import element
from rdkit import Chem

from bonafide.utils.helper_functions_chemistry import (
    bind_smiles_with_xyz,
    from_periodic_table,
    get_atom_bond_mapping_dicts,
    get_charge_from_mol_object,
    get_molecular_formula,
    get_ring_classification,
    get_symmetric_atom_sites,
)

SMILES_READING_PARAMS = Chem.SmilesParserParams()
SMILES_READING_PARAMS.removeHs = False
SMILES_READING_PARAMS.sanitize = True

##################################################
# Tests for the bind_smiles_with_xyz() function. #
##################################################


@pytest.fixture(
    params=[
        ("van_der_waals", True),
        ("van_der_waals", False),
        ("hueckel", True),
        ("hueckel", False),
        ("connect_the_dots", True),
        ("connect_the_dots", False),
    ],
)
def conn_method_align(request: pytest.FixtureRequest) -> Tuple[str, bool]:
    return request.param


@pytest.mark.bind_smiles_with_xyz
def test_bind_smiles_with_xyz(
    fetch_data_file: Callable[[str], str], conn_method_align: Tuple[str, bool]
) -> None:
    """Test for the ``bind_smiles_with_xyz()`` function: everything works as expected with
    different methods and with/without alignment for a neutral organic molecule.
    """
    # Get parameters
    connectivity_method, do_alignment = conn_method_align

    # Generate RDKit mol object from SMILES string
    smiles_mol = Chem.MolFromSmiles(
        "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])c([H])"
        "c1Cl)C([H])([H])C2([H])[H]",
        params=SMILES_READING_PARAMS,
    )
    assert smiles_mol.GetNumBonds() == 39

    assert smiles_mol.GetAtomWithIdx(0).GetSymbol() == "H"
    assert smiles_mol.GetAtomWithIdx(2).GetSymbol() == "S"
    assert smiles_mol.GetAtomWithIdx(10).GetSymbol() == "N"
    assert smiles_mol.GetAtomWithIdx(16).GetSymbol() == "C"
    assert smiles_mol.GetAtomWithIdx(30).GetSymbol() == "Cl"

    assert smiles_mol.GetBondWithIdx(4).GetBondTypeAsDouble() == 1.5
    assert smiles_mol.GetBondWithIdx(9).GetBondTypeAsDouble() == 1.0
    assert smiles_mol.GetBondWithIdx(13).GetBondTypeAsDouble() == 2.0

    # Generate RDKit mol object from XYZ file
    xyz_mol = Chem.MolFromXYZFile(fetch_data_file(file_name="clopidogrel-conf_01.xyz"))
    assert xyz_mol.GetNumBonds() == 0
    assert xyz_mol.GetNumAtoms() == 37
    assert xyz_mol.GetNumAtoms() == smiles_mol.GetNumAtoms()

    # Do binding
    new_mol, error_message = bind_smiles_with_xyz(
        smiles_mol=smiles_mol,
        xyz_mol=xyz_mol,
        align=do_alignment,
        connectivity_method=connectivity_method,
        covalent_radius_factor=1.3,
        charge=0,
    )
    assert new_mol is not None
    assert error_message is None
    assert new_mol.GetNumBonds() == 39

    # Check atoms (indices dependent on alignment)
    if do_alignment is True:
        assert new_mol.GetAtomWithIdx(32).GetSymbol() == "H"
        assert new_mol.GetAtomWithIdx(12).GetSymbol() == "S"
        assert new_mol.GetAtomWithIdx(4).GetSymbol() == "N"
        assert new_mol.GetAtomWithIdx(0).GetSymbol() == "C"
        assert new_mol.GetAtomWithIdx(19).GetSymbol() == "Cl"
    else:
        assert new_mol.GetAtomWithIdx(0).GetSymbol() == "H"
        assert new_mol.GetAtomWithIdx(2).GetSymbol() == "S"
        assert new_mol.GetAtomWithIdx(10).GetSymbol() == "N"
        assert new_mol.GetAtomWithIdx(16).GetSymbol() == "C"
        assert new_mol.GetAtomWithIdx(30).GetSymbol() == "Cl"

    # Check atoms (indices not dependent on alignment)
    assert new_mol.GetBondWithIdx(4).GetBondTypeAsDouble() == 1.5
    assert new_mol.GetBondWithIdx(9).GetBondTypeAsDouble() == 1.0
    assert new_mol.GetBondWithIdx(13).GetBondTypeAsDouble() == 2.0


@pytest.mark.bind_smiles_with_xyz
def test_bind_smiles_with_xyz2(
    fetch_data_file: Callable[[str], str], conn_method_align: Tuple[str, bool]
) -> None:
    """Test for the ``bind_smiles_with_xyz()`` function: everything works as expected with
    different methods and with/without alignment for an open-shell Cr(III) complex.
    """
    # Get parameters
    connectivity_method, do_alignment = conn_method_align

    # Generate RDKit mol object from SMILES string
    smiles_mol = Chem.MolFromSmiles(
        "Cl[Cr+]12(Cl)(<-N(C(C(P->1([H])[H])([H])[H])([H])[H])([H])[H])<-N(C(C([H])([H])P->2"
        "(C([H])([H])[H])C([H])([H])[H])([H])[H])([H])[H]",
        params=SMILES_READING_PARAMS,
    )
    assert smiles_mol.GetNumBonds() == 34

    assert smiles_mol.GetAtomWithIdx(0).GetSymbol() == "Cl"
    assert smiles_mol.GetAtomWithIdx(1).GetSymbol() == "Cr"
    assert smiles_mol.GetAtomWithIdx(15).GetSymbol() == "N"
    assert smiles_mol.GetAtomWithIdx(20).GetSymbol() == "P"

    assert smiles_mol.GetBondWithIdx(14).GetBondTypeAsDouble() == 1.0
    assert smiles_mol.GetBondWithIdx(33).GetBondTypeAsDouble() == 1.0

    # Generate RDKit mol object from XYZ file
    xyz_mol = Chem.MolFromXYZFile(fetch_data_file(file_name="Cr_III_cation-conf_00.xyz"))
    assert xyz_mol.GetNumBonds() == 0
    assert xyz_mol.GetNumAtoms() == 33
    assert xyz_mol.GetNumAtoms() == smiles_mol.GetNumAtoms()

    # Do binding
    new_mol, error_message = bind_smiles_with_xyz(
        smiles_mol=smiles_mol,
        xyz_mol=xyz_mol,
        align=do_alignment,
        connectivity_method=connectivity_method,
        covalent_radius_factor=1.3,
        charge=1,
    )
    assert new_mol is not None
    assert error_message is None
    assert new_mol.GetNumBonds() == 34

    # Check atoms (indices dependent on alignment)
    if do_alignment is True:
        assert new_mol.GetAtomWithIdx(8).GetSymbol() == "Cl"
        assert new_mol.GetAtomWithIdx(6).GetSymbol() == "Cr"
        assert new_mol.GetAtomWithIdx(5).GetSymbol() == "N"
        assert new_mol.GetAtomWithIdx(12).GetSymbol() == "P"
    else:
        assert new_mol.GetAtomWithIdx(0).GetSymbol() == "Cl"
        assert new_mol.GetAtomWithIdx(1).GetSymbol() == "Cr"
        assert new_mol.GetAtomWithIdx(15).GetSymbol() == "N"
        assert new_mol.GetAtomWithIdx(20).GetSymbol() == "P"

    # Check atoms (indices not dependent on alignment)
    assert new_mol.GetBondWithIdx(14).GetBondTypeAsDouble() == 1.0
    assert new_mol.GetBondWithIdx(33).GetBondTypeAsDouble() == 1.0


@pytest.mark.bind_smiles_with_xyz
def test_bind_smiles_with_xyz3(
    fetch_data_file: Callable[[str], str], conn_method_align: Tuple[str, bool]
) -> None:
    """Test for the ``bind_smiles_with_xyz()`` function: everything works as expected with
    different methods and with/without alignment for an organic radical cation.
    """
    # Get parameters
    connectivity_method, do_alignment = conn_method_align

    # Generate RDKit mol object from SMILES string
    smiles_mol = Chem.MolFromSmiles(
        "[H]C1=C(OC([H])([H])[H])[N+](C(N1C([H])([H])[H])=S)[H]", params=SMILES_READING_PARAMS
    )
    assert smiles_mol.GetNumBonds() == 17

    assert smiles_mol.GetAtomWithIdx(3).GetSymbol() == "O"
    assert smiles_mol.GetAtomWithIdx(8).GetSymbol() == "N"
    assert smiles_mol.GetAtomWithIdx(15).GetSymbol() == "S"

    assert smiles_mol.GetAtomWithIdx(8).GetNumRadicalElectrons() == 1
    assert smiles_mol.GetAtomWithIdx(8).GetFormalCharge() == 1

    assert smiles_mol.GetBondWithIdx(3).GetBondTypeAsDouble() == 1.0
    assert smiles_mol.GetBondWithIdx(14).GetBondTypeAsDouble() == 2.0

    # Generate RDKit mol object from XYZ file
    xyz_mol = Chem.MolFromXYZFile(fetch_data_file(file_name="radical_cation-conf_01.xyz"))
    assert xyz_mol.GetNumBonds() == 0
    assert xyz_mol.GetNumAtoms() == 17
    assert xyz_mol.GetNumAtoms() == smiles_mol.GetNumAtoms()

    # Do binding
    new_mol, error_message = bind_smiles_with_xyz(
        smiles_mol=smiles_mol,
        xyz_mol=xyz_mol,
        align=do_alignment,
        connectivity_method=connectivity_method,
        covalent_radius_factor=1.3,
        charge=1,
    )
    assert new_mol is not None
    assert error_message is None
    assert new_mol.GetNumBonds() == 17

    # Check atoms (indices dependent on alignment)
    if do_alignment is True:
        assert new_mol.GetAtomWithIdx(4).GetSymbol() == "O"
        assert new_mol.GetAtomWithIdx(2).GetSymbol() == "N"
        assert new_mol.GetAtomWithIdx(0).GetSymbol() == "S"

        assert new_mol.GetAtomWithIdx(2).GetNumRadicalElectrons() == 1
        assert new_mol.GetAtomWithIdx(2).GetFormalCharge() == 1

    else:
        assert new_mol.GetAtomWithIdx(3).GetSymbol() == "O"
        assert new_mol.GetAtomWithIdx(8).GetSymbol() == "N"
        assert new_mol.GetAtomWithIdx(15).GetSymbol() == "S"

        assert new_mol.GetAtomWithIdx(8).GetNumRadicalElectrons() == 1
        assert new_mol.GetAtomWithIdx(8).GetFormalCharge() == 1

    # Check atoms (indices not dependent on alignment)
    assert new_mol.GetBondWithIdx(3).GetBondTypeAsDouble() == 1.0
    assert new_mol.GetBondWithIdx(14).GetBondTypeAsDouble() == 2.0


@pytest.mark.bind_smiles_with_xyz
def test_bind_smiles_with_xyz4(
    fetch_data_file: Callable[[str], str], conn_method_align: Tuple[str, bool]
) -> None:
    """Test for the ``bind_smiles_with_xyz()`` function: everything works as expected with
    different methods and with/without alignment for a neutral organic molecule. SMILES
    string is atom-mapped.
    """
    # Get parameters
    connectivity_method, do_alignment = conn_method_align

    # Generate RDKit mol object from SMILES string
    smiles_mol = Chem.MolFromSmiles(
        "c12[c:1]([s:2][c:3]([H:4])[c:5]1[H:6])[C:7]([H:8])([H:36])[C:9]([H:10])([H:11])[N:12]"
        "([C@@:16]([c:17]1[c:18]([H:27])[c:19]([H:20])[c:21]([H:26])[c:22]([H:25])[c:23]1[Cl:24])"
        "([C:28](=[O:29])[O:30][C:31]([H:32])([H:33])[H:34])[H:35])[C:13]2([H:14])[H:15]",
        params=SMILES_READING_PARAMS,
    )
    assert smiles_mol.GetNumBonds() == 39

    assert smiles_mol.GetAtomWithIdx(2).GetSymbol() == "S"
    assert smiles_mol.GetAtomWithIdx(3).GetSymbol() == "C"
    assert smiles_mol.GetAtomWithIdx(13).GetSymbol() == "N"
    assert smiles_mol.GetAtomWithIdx(25).GetSymbol() == "Cl"
    assert smiles_mol.GetAtomWithIdx(30).GetSymbol() == "H"

    assert smiles_mol.GetBondWithIdx(4).GetBondTypeAsDouble() == 1.5
    assert smiles_mol.GetBondWithIdx(20).GetBondTypeAsDouble() == 1.0
    assert smiles_mol.GetBondWithIdx(26).GetBondTypeAsDouble() == 2.0

    # Generate RDKit mol object from XYZ file
    xyz_mol = Chem.MolFromXYZFile(fetch_data_file(file_name="clopidogrel-conf_01.xyz"))
    assert xyz_mol.GetNumBonds() == 0
    assert xyz_mol.GetNumAtoms() == 37
    assert xyz_mol.GetNumAtoms() == smiles_mol.GetNumAtoms()

    # Do binding
    new_mol, error_message = bind_smiles_with_xyz(
        smiles_mol=smiles_mol,
        xyz_mol=xyz_mol,
        align=do_alignment,
        connectivity_method=connectivity_method,
        covalent_radius_factor=1.3,
        charge=0,
    )
    assert new_mol is not None
    assert error_message is None
    assert new_mol.GetNumBonds() == 39

    # Check atoms (indices dependent on alignment)
    if do_alignment is True:
        assert new_mol.GetAtomWithIdx(32).GetSymbol() == "H"
        assert new_mol.GetAtomWithIdx(12).GetSymbol() == "S"
        assert new_mol.GetAtomWithIdx(4).GetSymbol() == "N"
        assert new_mol.GetAtomWithIdx(0).GetSymbol() == "C"
        assert new_mol.GetAtomWithIdx(19).GetSymbol() == "Cl"
    else:
        assert new_mol.GetAtomWithIdx(2).GetSymbol() == "S"
        assert new_mol.GetAtomWithIdx(3).GetSymbol() == "C"
        assert new_mol.GetAtomWithIdx(13).GetSymbol() == "N"
        assert new_mol.GetAtomWithIdx(25).GetSymbol() == "Cl"
        assert new_mol.GetAtomWithIdx(30).GetSymbol() == "H"

    # Check atoms (indices not dependent on alignment)
    assert new_mol.GetBondWithIdx(4).GetBondTypeAsDouble() == 1.5
    assert new_mol.GetBondWithIdx(20).GetBondTypeAsDouble() == 1.0
    assert new_mol.GetBondWithIdx(26).GetBondTypeAsDouble() == 2.0


@pytest.mark.bind_smiles_with_xyz
def test_bind_smiles_with_xyz5(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``bind_smiles_with_xyz()`` function: fails due to different number of bonds
    after determining the atom connectivity.
    """
    # Generate RDKit mol object from SMILES string
    smiles_mol = Chem.MolFromSmiles(
        "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])c([H])"
        "c1Cl)C([H])([H])C2([H])[H]",
        params=SMILES_READING_PARAMS,
    )
    assert smiles_mol.GetNumBonds() == 39

    # Generate RDKit mol object from XYZ file
    xyz_mol = Chem.MolFromXYZFile(fetch_data_file(file_name="clopidogrel-conf_01.xyz"))
    assert xyz_mol.GetNumBonds() == 0
    assert xyz_mol.GetNumAtoms() == 37
    assert xyz_mol.GetNumAtoms() == smiles_mol.GetNumAtoms()

    # Do binding
    new_mol, error_message = bind_smiles_with_xyz(
        smiles_mol=smiles_mol,
        xyz_mol=xyz_mol,
        align=True,
        connectivity_method="van_der_waals",
        covalent_radius_factor=2.3,
        charge=0,
    )
    assert new_mol is None
    assert error_message.startswith(
        "the number of atom connectivities (127) determined in the RDKit mol object generated "
        "from the XYZ block does not match"
    )


#########################################################
# Tests for the get_atom_bond_mapping_dicts() function. #
#########################################################


@pytest.mark.get_atom_bond_mapping_dicts
@pytest.mark.parametrize(
    "smiles, expected_mapping_dict_atoms, expected_mapping_dict_bonds, expected_canonical_smiles",
    [
        # Canonical SMILES string without H
        ("OCF", {0: 0, 1: 1, 2: 2}, {0: 0, 1: 1}, "OCF"),
        # Canonical SMILES string without H
        (
            "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
            {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                6: 6,
                7: 7,
                8: 8,
                9: 9,
                10: 10,
                11: 11,
                12: 12,
                13: 13,
                14: 14,
                15: 15,
                16: 16,
                17: 17,
                18: 18,
                19: 19,
                20: 20,
            },
            {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                6: 6,
                7: 7,
                8: 8,
                9: 9,
                10: 10,
                11: 11,
                12: 12,
                13: 13,
                14: 14,
                15: 15,
                16: 16,
                17: 17,
                18: 18,
                19: 19,
                20: 20,
                21: 21,
                22: 22,
            },
            "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
        ),
        # Non-canonical SMILES string without H
        (
            "COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O",
            {
                0: 0,
                1: 1,
                2: 2,
                3: 20,
                4: 3,
                5: 13,
                6: 14,
                7: 15,
                8: 16,
                9: 17,
                10: 18,
                11: 19,
                12: 4,
                13: 5,
                14: 6,
                15: 7,
                16: 12,
                17: 11,
                18: 10,
                19: 8,
                20: 9,
            },
            {
                0: 0,
                1: 1,
                2: 19,
                3: 2,
                4: 12,
                5: 13,
                6: 14,
                7: 15,
                8: 16,
                9: 17,
                10: 18,
                11: 3,
                12: 4,
                13: 5,
                14: 6,
                15: 21,
                16: 11,
                17: 10,
                18: 9,
                19: 8,
                20: 22,
                21: 20,
                22: 7,
            },
            "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
        ),
        # Canonical SMILES string with H
        (
            "[H]OC([H])([H])F",
            {
                0: 1,
                1: 2,
                2: 5,
                3: 0,
                4: 3,
                5: 4,
            },
            {
                0: 1,
                1: 4,
                2: 0,
                3: 2,
                4: 3,
            },
            "OCF",
        ),
        # Non-canonical SMILES string with H
        (
            "O(C([H])([H])F)[H]",
            {
                0: 0,
                1: 1,
                2: 4,
                3: 5,
                4: 2,
                5: 3,
            },
            {
                0: 0,
                1: 3,
                2: 4,
                3: 1,
                4: 2,
            },
            "OCF",
        ),
        # Non-canonical SMILES string with H
        (
            "O1C2(C(N(C(C(C3([H])C(C3([H])[H])([H])[H])([H])[H])([H])[H])C(C1(F)F)([H])[H])"
            "([H])[H])C(C([H])=C(C(C2([H])[H])=O)[H])([H])[H]",
            {
                0: 34,
                1: 30,
                2: 29,
                3: 27,
                4: 26,
                5: 1,
                6: 31,
                7: 2,
                8: 3,
                9: 4,
                10: 5,
                11: 6,
                12: 8,
                13: 9,
                14: 18,
                15: 19,
                16: 20,
                17: 21,
                18: 0,
                19: 35,
                20: 28,
                21: 36,
                22: 37,
                23: 32,
                24: 33,
                25: 24,
                26: 25,
                27: 16,
                28: 17,
                29: 14,
                30: 15,
                31: 7,
                32: 12,
                33: 13,
                34: 10,
                35: 11,
                36: 22,
                37: 23,
            },
            {
                0: 33,
                1: 29,
                2: 28,
                3: 26,
                4: 25,
                5: 38,
                6: 1,
                7: 2,
                8: 3,
                9: 4,
                10: 5,
                11: 7,
                12: 8,
                13: 17,
                14: 18,
                15: 19,
                16: 20,
                17: 37,
                18: 30,
                19: 39,
                20: 0,
                21: 34,
                22: 27,
                23: 35,
                24: 36,
                25: 31,
                26: 32,
                27: 23,
                28: 24,
                29: 15,
                30: 16,
                31: 13,
                32: 14,
                33: 6,
                34: 11,
                35: 12,
                36: 9,
                37: 10,
                38: 21,
                39: 22,
            },
            "O=C1C=CCC2(C1)CN(CCC1CC1)CC(F)(F)O2",
        ),
    ],
)
def test_get_atom_bond_mapping_dicts(
    smiles: str,
    expected_mapping_dict_atoms: Dict[int, int],
    expected_mapping_dict_bonds: Dict[int, int],
    expected_canonical_smiles: str,
) -> None:
    """Test for the ``get_atom_bond_mapping_dicts()`` function."""
    # Generate RDKit mol object
    mol = Chem.MolFromSmiles(smiles, params=SMILES_READING_PARAMS)

    # Calculate mapping dicts and canonical SMILES
    (
        mapping_dict_atoms,
        mapping_dict_bonds,
        canonical_smiles,
    ) = get_atom_bond_mapping_dicts(mol=mol)

    # Check results
    assert mapping_dict_atoms == expected_mapping_dict_atoms
    assert mapping_dict_bonds == expected_mapping_dict_bonds
    assert canonical_smiles == expected_canonical_smiles


#################################################
# Tests for the get_charge_from_mol_object() function. #
#################################################


@pytest.mark.get_charge_from_mol_object
@pytest.mark.parametrize(
    "smiles, expected_charge",
    [
        (
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])"
            "c([H])c1Cl)C([H])([H])C2([H])[H]",
            0,
        ),
        (
            "O=C1C=CCC2(CN(CCC3CC3)CC(F)(F)O2)C1",
            0,
        ),
        ("CC[C@H]([NH3+])C([O-])=O", 0),
        (
            "C1(=S)[N+]([H])C(OC([H])([H])[H])=C(N1C([H])([H])[H])[H]",
            1,
        ),
        (
            "[H]C([H])([O-])C([H])([H])[C@@]([H])(C(=O)[O-])[N+]([H])([H])[H]",
            -1,
        ),
        ("O=C([O-])CN(CC(=O)[O-])C(=O)[O-]", -3),
        (
            "[H]C(P1(C([H])([H])[H])->[Cr+]2(Cl)(<-N([H])([H])C([H])(C(P->2([H])[H])([H])[H])[H])"
            "(<-N(C(C1([H])[H])([H])[H])([H])[H])Cl)([H])[H]",
            1,
        ),
    ],
)
def test_get_charge_from_mol_object(smiles: str, expected_charge: int) -> None:
    """Test for the ``get_charge_from_mol_object()`` function."""
    # Generate RDKit mol object
    mol = Chem.MolFromSmiles(smiles, params=SMILES_READING_PARAMS)

    # Calculate multiplicity
    assert get_charge_from_mol_object(mol) == expected_charge


#################################################
# Tests for the from_periodic_table() function. #
#################################################


@pytest.mark.from_periodic_table
def test_from_periodic_table1() -> None:
    """Test for the ``from_periodic_table()`` function: empty periodic table"""
    _el = "Br"
    periodic_table, element_data = from_periodic_table(periodic_table={}, element_symbol=_el)
    assert _el in periodic_table

    _data = element(_el)
    assert periodic_table[_el] == _data
    assert element_data == _data


@pytest.mark.from_periodic_table
def test_from_periodic_table2() -> None:
    """Test for the ``from_periodic_table()`` function: non-empty periodic table"""
    _el_existing = "C"
    _el_new = "N"
    periodic_table = {_el_existing: element(_el_existing)}

    periodic_table, element_data = from_periodic_table(
        periodic_table=periodic_table, element_symbol=_el_new
    )
    assert len(periodic_table) == 2
    assert _el_existing in periodic_table
    assert _el_new in periodic_table

    _data_new = element(_el_new)
    assert periodic_table[_el_new] == _data_new
    assert element_data == _data_new


@pytest.mark.from_periodic_table
def test_from_periodic_table3() -> None:
    """Test for the ``from_periodic_table()`` function: element already in periodic table"""
    _el_existing = "O"
    periodic_table = {_el_existing: element(_el_existing)}

    periodic_table, element_data = from_periodic_table(
        periodic_table=periodic_table, element_symbol=_el_existing
    )
    assert len(periodic_table) == 1
    assert _el_existing in periodic_table

    _data_existing = element(_el_existing)
    assert periodic_table[_el_existing] == _data_existing
    assert element_data == _data_existing


#####################################################
# Tests for the get_ring_classification() function. #
#####################################################


@pytest.mark.get_ring_classification
@pytest.mark.parametrize(
    "smiles, indices, idx_type, expected_ring_class",
    [
        (
            "[H]c1sc2c(c1[H])C([H])([H])N([C@]([H])(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])c"
            "([H])c1Cl)C([H])([H])C2([H])[H]",
            [20, 21, 23, 25, 27, 29],
            "atom",
            "aromatic_carbocycle",
        ),
        (
            "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
            [17, 18, 19, 15, 16],
            "atom",
            "aromatic_heterocycle",
        ),
        (
            "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
            [16, 17, 15, 22, 18],
            "bond",
            "aromatic_heterocycle",
        ),
        (
            "O1C2(C(N(C(C(C3([H])C(C3([H])[H])([H])[H])([H])[H])([H])[H])C(C1(F)F)([H])[H])([H])"
            "[H])C(C([H])=C(C(C2([H])[H])=O)[H])([H])[H]",
            [30, 29, 28, 26, 25, 38],
            "bond",
            "nonaromatic_carbocycle",
        ),
        (
            "O=C1C=CCC2(C1)CN(CCC1CC1)CC(F)(F)O2",
            [11, 12, 13],
            "atom",
            "nonaromatic_carbocycle",
        ),
        (
            "O=C1C=CCC2(CN(CCC3CC3)CC(F)(F)O2)C1",
            [17, 14, 13, 7, 6, 5],
            "atom",
            "nonaromatic_heterocycle",
        ),
        (
            "O=C1C=CCC2(CN(CCC3CC3)CC(F)(F)O2)C1",
            [5, 6, 12, 13, 16, 19],
            "bond",
            "nonaromatic_heterocycle",
        ),
        # "Non-cyclic" atoms
        (
            "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
            [0, 1, 2, 3],
            "atom",
            "None",
        ),
        # "Non-cyclic" bonds
        (
            "O=C1C=CCC2(C1)CN(CCC1CC1)CC(F)(F)O2",
            [8, 9, 10, 11, 12],
            "bond",
            "None",
        ),
    ],
)
def test_get_ring_classification(
    smiles: str, indices: List[int], idx_type: str, expected_ring_class: str
) -> None:
    """Test for the ``get_ring_classification()`` function."""
    # Generate RDKit mol object
    mol = Chem.MolFromSmiles(smiles, params=SMILES_READING_PARAMS)

    # Calculate ring classification
    assert get_ring_classification(mol, indices, idx_type) == expected_ring_class


###################################################
# Tests for the get_molecular_formula() function. #
###################################################


@pytest.mark.get_molecular_formula
@pytest.mark.parametrize(
    "smiles, expected_formula",
    [
        ("CCO", "C2O1"),
        ("[H]OC([H])([H])C([H])([H])[H]", "C2H6O1"),
        ("[H]OC([H])([H])F", "C1F1H3O1"),
        ("COC([C@@H](N1CCC2=C(C1)C=CS2)C3=CC=CC=C3Cl)=O", "C16Cl1N1O2S1"),
        (
            "c12c(sc([H])c1[H])C([H])(C([H])([H])N(C2([H])[H])[C@@](c1c(c([H])c(c(c1Cl)[H])[H])[H])(C(=O)OC([H])([H])[H])[H])[H]",
            "C16Cl1H16N1O2S1",
        ),
    ],
)
def test_get_molecular_formula(smiles: str, expected_formula: str) -> None:
    """Test for the ``get_molecular_formula()`` function."""
    # Generate RDKit mol object
    mol = Chem.MolFromSmiles(smiles, params=SMILES_READING_PARAMS)

    # Calculate molecular formula
    assert get_molecular_formula(mol) == expected_formula


######################################################
# Tests for the get_symmetric_atom_sites() function. #
######################################################

_params = [
    "smiles",
    "expected_output",
    "include_chirality",
    "include_isotopes",
    "include_atom_maps",
    "include_chiral_presence",
    "consider_resonance",
    "resonance_ALLOW_CHARGE_SEPARATION",
    "resonance_ALLOW_INCOMPLETE_OCTETS",
    "resonance_KEKULE_ALL",
    "resonance_UNCONSTRAINED_ANIONS",
    "resonance_UNCONSTRAINED_CATIONS",
]


@pytest.mark.get_symmetric_atom_sites
@pytest.mark.parametrize(
    ", ".join(_params),
    [
        (
            "O=C([O-])C1=CC=CC=C1",
            {0: [0, 2], 1: [1], 3: [3], 4: [4, 8], 5: [5, 7], 6: [6]},
            True,  # include_chirality
            True,  # include_isotopes
            False,  # include_atom_maps
            False,  # include_chiral_presence
            True,  # consider_resonance
            False,  # resonance_ALLOW_CHARGE_SEPARATION
            False,  # resonance_ALLOW_INCOMPLETE_OCTETS
            False,  # resonance_KEKULE_ALL
            False,  # resonance_UNCONSTRAINED_ANIONS
            False,  # resonance_UNCONSTRAINED_CATIONS
        ),
        (
            "O=C([O-])C1=CC=CC=C1",
            {0: [0], 1: [1], 2: [2], 3: [3], 4: [4, 8], 5: [5, 7], 6: [6]},
            True,  # include_chirality
            True,  # include_isotopes
            False,  # include_atom_maps
            False,  # include_chiral_presence
            False,  # consider_resonance
            False,  # resonance_ALLOW_CHARGE_SEPARATION
            False,  # resonance_ALLOW_INCOMPLETE_OCTETS
            False,  # resonance_KEKULE_ALL
            False,  # resonance_UNCONSTRAINED_ANIONS
            False,  # resonance_UNCONSTRAINED_CATIONS
        ),
        (
            "[H]c1c([H])c(C([H])([H])c2c([H])c([H])c([N+](=O)[O-])c(F)c2[N+](=O)[O-])c([N+](=O)"
            "[O-])c(F)c1[N+](=O)[O-]",
            {
                0: [0, 12],
                1: [1, 11],
                2: [2, 9],
                3: [3, 10],
                4: [4, 8],
                5: [5],
                6: [6, 7],
                13: [13, 29],
                14: [14, 30],
                15: [15, 16, 31, 32],
                17: [17, 27],
                18: [18, 28],
                19: [19, 23],
                20: [20, 24],
                21: [21, 22, 25, 26],
            },
            True,  # include_chirality
            True,  # include_isotopes
            False,  # include_atom_maps
            False,  # include_chiral_presence
            True,  # consider_resonance
            False,  # resonance_ALLOW_CHARGE_SEPARATION
            False,  # resonance_ALLOW_INCOMPLETE_OCTETS
            False,  # resonance_KEKULE_ALL
            False,  # resonance_UNCONSTRAINED_ANIONS
            False,  # resonance_UNCONSTRAINED_CATIONS
        ),
        (
            "C[C@@H](F)N/C(C1=CC=CC=C1)=[NH+]/[C@@H](F)C",  # meso form
            {
                0: [0, 14],
                1: [1, 12],
                2: [2, 13],
                3: [3, 11],
                4: [4],
                5: [5],
                6: [6, 10],
                7: [7, 9],
                8: [8],
            },
            True,  # include_chirality
            True,  # include_isotopes
            False,  # include_atom_maps
            False,  # include_chiral_presence
            True,  # consider_resonance
            False,  # resonance_ALLOW_CHARGE_SEPARATION
            False,  # resonance_ALLOW_INCOMPLETE_OCTETS
            False,  # resonance_KEKULE_ALL
            False,  # resonance_UNCONSTRAINED_ANIONS
            False,  # resonance_UNCONSTRAINED_CATIONS
        ),
        (
            "C[C@@H](F)N/C(C1=CC=CC=C1)=[NH+]/[C@@H](F)C",  # meso form
            {
                0: [0, 14],
                1: [1, 12],
                2: [2, 13],
                3: [3, 11],
                4: [4],
                5: [5],
                6: [6, 10],
                7: [7, 9],
                8: [8],
            },
            False,  # include_chirality
            True,  # include_isotopes
            False,  # include_atom_maps
            False,  # include_chiral_presence
            True,  # consider_resonance
            False,  # resonance_ALLOW_CHARGE_SEPARATION
            False,  # resonance_ALLOW_INCOMPLETE_OCTETS
            False,  # resonance_KEKULE_ALL
            False,  # resonance_UNCONSTRAINED_ANIONS
            False,  # resonance_UNCONSTRAINED_CATIONS
        ),
        (
            "C[C@@H](F)N/C(C1=CC=CC=C1)=[NH+]/[C@@H](F)C",  # meso form
            {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [5],
                6: [6, 10],
                7: [7, 9],
                8: [8],
                11: [11],
                12: [12],
                13: [13],
                14: [14],
            },
            False,  # include_chirality
            True,  # include_isotopes
            False,  # include_atom_maps
            False,  # include_chiral_presence
            False,  # consider_resonance
            False,  # resonance_ALLOW_CHARGE_SEPARATION
            False,  # resonance_ALLOW_INCOMPLETE_OCTETS
            False,  # resonance_KEKULE_ALL
            False,  # resonance_UNCONSTRAINED_ANIONS
            False,  # resonance_UNCONSTRAINED_CATIONS
        ),
        (
            "C[C@H](F)N/C(C1=CC=CC=C1)=[NH+]/[C@@H](F)C",  # chiral form
            {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [5],
                6: [6, 10],
                7: [7, 9],
                8: [8],
                11: [11],
                12: [12],
                13: [13],
                14: [14],
            },
            True,  # include_chirality
            True,  # include_isotopes
            False,  # include_atom_maps
            False,  # include_chiral_presence
            True,  # consider_resonance
            False,  # resonance_ALLOW_CHARGE_SEPARATION
            False,  # resonance_ALLOW_INCOMPLETE_OCTETS
            False,  # resonance_KEKULE_ALL
            False,  # resonance_UNCONSTRAINED_ANIONS
            False,  # resonance_UNCONSTRAINED_CATIONS
        ),
        (
            "C[C@H](F)N/C(C1=CC=CC=C1)=[NH+]/[C@@H](F)C",  # chiral form
            {
                0: [0, 14],
                1: [1, 12],
                2: [2, 13],
                3: [3, 11],
                4: [4],
                5: [5],
                6: [6, 10],
                7: [7, 9],
                8: [8],
            },
            False,  # include_chirality
            True,  # include_isotopes
            False,  # include_atom_maps
            False,  # include_chiral_presence
            True,  # consider_resonance
            False,  # resonance_ALLOW_CHARGE_SEPARATION
            False,  # resonance_ALLOW_INCOMPLETE_OCTETS
            False,  # resonance_KEKULE_ALL
            False,  # resonance_UNCONSTRAINED_ANIONS
            False,  # resonance_UNCONSTRAINED_CATIONS
        ),
        (
            "CC(S([O-])=O)C(S(=O)([O-])=O)CCC(P([O-])=O)C(P([O-])([O-])=O)C",
            {
                0: [0],
                1: [1],
                2: [2],
                3: [3, 4],
                5: [5],
                6: [6],
                7: [7, 8, 9],
                10: [10],
                11: [11],
                12: [12],
                13: [13],
                14: [14, 15],
                16: [16],
                17: [17],
                18: [18, 19, 20],
                21: [21],
            },
            True,  # include_chirality
            True,  # include_isotopes
            False,  # include_atom_maps
            False,  # include_chiral_presence
            True,  # consider_resonance
            False,  # resonance_ALLOW_CHARGE_SEPARATION
            False,  # resonance_ALLOW_INCOMPLETE_OCTETS
            False,  # resonance_KEKULE_ALL
            False,  # resonance_UNCONSTRAINED_ANIONS
            False,  # resonance_UNCONSTRAINED_CATIONS
        ),
        (
            "CC(S([O-])=O)C(S(=O)([O-])=O)CCC(P([O-])=O)C(P([O-])([O-])=O)C",
            {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [5],
                6: [6],
                7: [7, 9],
                8: [8],
                10: [10],
                11: [11],
                12: [12],
                13: [13],
                14: [14],
                15: [15],
                16: [16],
                17: [17],
                18: [18, 19],
                20: [20],
                21: [21],
            },
            True,  # include_chirality
            True,  # include_isotopes
            False,  # include_atom_maps
            False,  # include_chiral_presence
            False,  # consider_resonance
            False,  # resonance_ALLOW_CHARGE_SEPARATION
            False,  # resonance_ALLOW_INCOMPLETE_OCTETS
            False,  # resonance_KEKULE_ALL
            False,  # resonance_UNCONSTRAINED_ANIONS
            False,  # resonance_UNCONSTRAINED_CATIONS
        ),
        (
            "[H]C([H])([H])C([H])(S(=O)[O-])C([H])(C([H])([H])C([H])([H])C([H])(C([H])(C([H])([H])"
            "[H])P(=O)([O-])[O-])P([H])(=O)[O-])S(=O)(=O)[O-]",
            {
                0: [0, 2, 3],
                1: [1],
                4: [4],
                5: [5],
                6: [6],
                7: [7, 8],
                9: [9],
                10: [10],
                11: [11],
                12: [12, 13],
                14: [14],
                15: [15, 16],
                17: [17],
                18: [18],
                19: [19],
                20: [20],
                21: [21],
                22: [22, 23, 24],
                25: [25],
                26: [26, 27, 28],
                29: [29],
                30: [30],
                31: [31, 32],
                33: [33],
                34: [34, 35, 36],
            },
            True,  # include_chirality
            True,  # include_isotopes
            False,  # include_atom_maps
            False,  # include_chiral_presence
            True,  # consider_resonance
            False,  # resonance_ALLOW_CHARGE_SEPARATION
            False,  # resonance_ALLOW_INCOMPLETE_OCTETS
            False,  # resonance_KEKULE_ALL
            False,  # resonance_UNCONSTRAINED_ANIONS
            False,  # resonance_UNCONSTRAINED_CATIONS
        ),
        (
            "[H]C([H])([H])C([H])(S(=O)[O-])C([H])(C([H])([H])C([H])([H])C([H])(C([H])(C([H])([H])"
            "[H])P(=O)([O-])[O-])P([H])(=O)[O-])S(=O)(=O)[O-]",
            {
                0: [0, 2, 3],
                1: [1],
                4: [4],
                5: [5],
                6: [6],
                7: [7],
                8: [8],
                9: [9],
                10: [10],
                11: [11],
                12: [12, 13],
                14: [14],
                15: [15, 16],
                17: [17],
                18: [18],
                19: [19],
                20: [20],
                21: [21],
                22: [22, 23, 24],
                25: [25],
                26: [26],
                27: [27, 28],
                29: [29],
                30: [30],
                31: [31],
                32: [32],
                33: [33],
                34: [34, 35],
                36: [36],
            },
            False,  # include_chirality
            False,  # include_isotopes
            False,  # include_atom_maps
            False,  # include_chiral_presence
            False,  # consider_resonance
            False,  # resonance_ALLOW_CHARGE_SEPARATION
            False,  # resonance_ALLOW_INCOMPLETE_OCTETS
            False,  # resonance_KEKULE_ALL
            False,  # resonance_UNCONSTRAINED_ANIONS
            False,  # resonance_UNCONSTRAINED_CATIONS
        ),
    ],
)
def test_get_symmetric_atom_sites(
    smiles: str,
    expected_output: Dict[int, List[int]],
    include_chirality: bool,
    include_isotopes: bool,
    include_atom_maps: bool,
    include_chiral_presence: bool,
    consider_resonance: bool,
    resonance_ALLOW_CHARGE_SEPARATION: bool,
    resonance_ALLOW_INCOMPLETE_OCTETS: bool,
    resonance_KEKULE_ALL: bool,
    resonance_UNCONSTRAINED_ANIONS: bool,
    resonance_UNCONSTRAINED_CATIONS: bool,
) -> None:
    """Test for the ``get_symmetric_atom_sites()`` function."""
    mol = Chem.MolFromSmiles(smiles, params=SMILES_READING_PARAMS)
    result = get_symmetric_atom_sites(
        mol=mol,
        include_chirality=include_chirality,
        include_isotopes=include_isotopes,
        include_atom_maps=include_atom_maps,
        include_chiral_presence=include_chiral_presence,
        consider_resonance=consider_resonance,
        resonance_ALLOW_CHARGE_SEPARATION=resonance_ALLOW_CHARGE_SEPARATION,
        resonance_ALLOW_INCOMPLETE_OCTETS=resonance_ALLOW_INCOMPLETE_OCTETS,
        resonance_KEKULE_ALL=resonance_KEKULE_ALL,
        resonance_UNCONSTRAINED_ANIONS=resonance_UNCONSTRAINED_ANIONS,
        resonance_UNCONSTRAINED_CATIONS=resonance_UNCONSTRAINED_CATIONS,
    )
    assert result == expected_output
