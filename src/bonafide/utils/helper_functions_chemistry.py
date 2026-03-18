"""Helper functions for chemistry-related operations."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from mendeleev import element
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from bonafide.utils.helper_functions import clean_up


def _get_renumbering_list(
    template: Chem.rdchem.Mol, to_be_renumbered: Chem.rdchem.Mol, invert: bool = False
) -> List[int]:
    """Get a renumbering list to reorder atoms in a molecule based on a template.

    Parameters
    ----------
    template : Chem.rdchem.Mol
        The RDKit molecule object that serves as the template for the atom order.
    to_be_renumbered : Chem.rdchem.Mol
        The RDKit molecule object that needs to be renumbered.
    invert : bool, optional
        Whether to invert the mapping dictionary, by default ``False``.

    Returns
    -------
    List[int]
        A list of integers representing the new atom order based on the template.
    """
    match = template.GetSubstructMatch(to_be_renumbered)

    if invert is True:
        mapping = {i: j for i, j in enumerate(match)}
    else:
        mapping = {j: i for i, j in enumerate(match)}

    mapping = dict(sorted(mapping.items()))
    renumbering_list = list(mapping.values())
    return renumbering_list


def _check_renumbering_list(renum_list: List[int], num_atoms: int) -> Optional[str]:
    """Check if a renumbering list is valid.

    Parameters
    ----------
    renum_list : List[int]
        The renumbering list to be checked.
    num_atoms : int
        The number of atoms in the respective molecule.

    Returns
    -------
    Optional[str]
        An error message if the renumbering list is invalid, otherwise ``None``.
    """
    _errmsg = None
    _len = len(renum_list)
    if _len != num_atoms:
        _errmsg = (
            "the substructure matching required for SMILES string attachment failed. The expected "
            f"number of matches ({num_atoms}) does not match the number of matches found "
            f"({_len}). Obtained matching atom indices list: {renum_list}. Check the input "
            "data or try different parameters for determining atom connectivity (input to "
            "the 'connectivity_method' and 'covalent_radius_factor' parameters)"
        )
    return _errmsg


def _set_atom_bond_properties(
    source_obj: Union[Chem.rdchem.Atom, Chem.rdchem.Bond],
    target_obj: Union[Chem.rdchem.Atom, Chem.rdchem.Bond],
) -> None:
    """Set properties from a source RDKit atom or bond object to a target RDKit atom or bond object.

    Parameters
    ----------
    source_obj : Union[Chem.rdchem.Atom, Chem.rdchem.Bond]
        The RDKit atom or bond object from which to transfer properties.
    target_obj : Union[Chem.rdchem.Atom, Chem.rdchem.Bond]
        The RDKit atom or bond object to which to transfer properties.

    Returns
    -------
    None
    """
    properties = source_obj.GetPropsAsDict()
    for property_name, value in properties.items():
        if type(value) == bool:
            target_obj.SetBoolProp(property_name, value)
        elif type(value) == int:
            target_obj.SetIntProp(property_name, value)
        elif type(value) == float:
            target_obj.SetDoubleProp(property_name, value)
        elif type(value) == str:
            target_obj.SetProp(property_name, value)


def _transfer_atom_bond_properties(
    source_mol: Chem.rdchem.Mol, target_mol: Chem.rdchem.Mol
) -> Chem.rdchem.Mol:
    """Transfer atom and bond properties from a source RDKit molecule object to a target RDKit
    molecule object.

    Parameters
    ----------
    source_mol : Chem.rdchem.Mol
        The RDKit molecule object from which to transfer properties.
    target_mol : Chem.rdchem.Mol
        The RDKit molecule object to which to transfer properties.

    Returns
    -------
    Chem.rdchem.Mol
        The target RDKit molecule object with transferred atom and bond properties.
    """
    # Transfer atom properties
    for source_atom, target_atom in zip(source_mol.GetAtoms(), target_mol.GetAtoms()):
        _set_atom_bond_properties(source_obj=source_atom, target_obj=target_atom)

    # Transfer bond properties
    for source_bond, target_bond in zip(source_mol.GetBonds(), target_mol.GetBonds()):
        _set_atom_bond_properties(source_obj=source_bond, target_obj=target_bond)

    return target_mol


def bind_smiles_with_xyz(
    smiles_mol: Chem.rdchem.Mol,
    xyz_mol: Chem.rdchem.Mol,
    align: bool,
    connectivity_method: str,
    covalent_radius_factor: float,
    charge: Optional[int],
) -> Tuple[Optional[Chem.rdchem.Mol], Optional[str]]:
    """Redefine an RDKit molecule object created from an XYZ file with a new RDKit molecule object
    created from a SMILES string.

    This allows to introduce the data on the chemical bonds defined in the SMILES string to the
    initial molecule object created from the XYZ file. The ``align`` parameter controls whether
    the atom order of the initial molecule object is maintained.

    The ``connectivity_method``, ``covalent_radius_factor``, and ``charge`` parameters define how
    the atom connectivity is determined in the RDKit molecule object created from the XYZ file.

    Parameters
    ----------
    smiles_mol : Chem.rdchem.Mol
        The RDKit molecule object created from a SMILES string.
    xyz_mol : Chem.rdchem.Mol
        The RDKit molecule object created from an XYZ file.
    align : bool
        If ``True``, the atom order of the ``xyz_mol`` will be maintained, if ``False``, the atom
        order of the ``smiles_mol`` will be applied.
    connectivity_method : str
        The name of the method that is used to determine atom connectivity. Available options are
        "connect_the_dots", "van_der_waals", and "hueckel".
    covalent_radius_factor : float
        A scaling factor that is applied to the covalent radii of the atoms when determining the
        atom connectivity with the van-der-Waals method.
    charge : Optional[int]
        The formal charge of the molecule, which is required when using the Hueckel method for
        determining atom connectivity.

    Returns
    -------
    Tuple[Optional[Chem.rdchem.Mol], Optional[str]]
        A tuple containing:

        * An RDKit molecule object containing the data from the ``smiles_mol`` applied to the
          ``xyz_mol``; ``None`` if the operation was unsuccessful.
        * An error message if the operation was unsuccessful, otherwise ``None``.
    """
    _errmsg = None

    # Prepare input to RDKit (which method to use)
    _use_hueckel = False
    _use_vdw = True
    if connectivity_method == "hueckel":
        _use_hueckel = True
    if connectivity_method == "connect_the_dots":
        _use_vdw = False

    # Get params to make substructure matches only based on atom type and connectivity
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True

    # Helper mol object for doing the substructure matching
    smiles_mol_helper = Chem.AdjustQueryProperties(smiles_mol, params)

    # Reset atom properties of the helper smiles mol to only have the atom types and
    # connectivity for substructure matching
    for atom in smiles_mol_helper.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        atom.SetHybridization(Chem.rdchem.HybridizationType.UNSPECIFIED)
        atom.SetIsAromatic(False)

    # Determine atom connectivity in the XYZ mol
    if _use_hueckel is True:
        rdDetermineBonds.DetermineConnectivity(mol=xyz_mol, useHueckel=_use_hueckel, charge=charge)
    else:
        rdDetermineBonds.DetermineConnectivity(
            mol=xyz_mol, covFactor=covalent_radius_factor, useVdw=_use_vdw
        )

    # Check if number of bonds matches
    _n_bonds_xyz = xyz_mol.GetNumBonds()
    _n_bonds_smiles = smiles_mol.GetNumBonds()
    if _n_bonds_xyz != _n_bonds_smiles:
        _errmsg = (
            f"the number of atom connectivities ({_n_bonds_xyz}) determined in the "
            "RDKit mol object generated from the XYZ block does not match the number of bonds "
            f"({_n_bonds_smiles}) in the RDKit mol object generated from the SMILES string. Check "
            "the input data or try different parameters for determining atom connectivity "
            "(input to the 'connectivity_method' and 'covalent_radius_factor' parameters)"
        )
        return None, _errmsg

    # Reference xyz_mol for storing the initial atom order
    xyz_mol_ref = Chem.Mol(xyz_mol)

    # Renumber the atoms of the XYZ mol to match the order of the smiles_mol
    xyz_mol = Chem.AdjustQueryProperties(xyz_mol, params)

    # In case of the Hueckel method, the substructure matching needs to be inverted as
    # it is not working the other way round
    if connectivity_method == "hueckel":
        renumbering_list = _get_renumbering_list(
            template=xyz_mol, to_be_renumbered=smiles_mol_helper, invert=True
        )
    else:
        renumbering_list = _get_renumbering_list(
            template=smiles_mol_helper, to_be_renumbered=xyz_mol
        )

    # Check if the renumbering list is valid
    error_message = _check_renumbering_list(
        renum_list=renumbering_list, num_atoms=xyz_mol.GetNumAtoms()
    )
    if error_message is not None:
        return None, error_message

    # Reorder the atoms of the xyz_mol to align with the order of the smiles_mol
    xyz_mol = Chem.RenumberAtoms(xyz_mol, renumbering_list)

    # Add the conformer to the real smiles mol and transfer the atom and bond properties
    smiles_mol.AddConformer(xyz_mol.GetConformer(0))
    smiles_mol = _transfer_atom_bond_properties(source_mol=xyz_mol, target_mol=smiles_mol)

    # Apply the atom order of xyz_mol_ref (restore the initial atom order)
    if align is True:
        renumbering_list = _get_renumbering_list(
            template=xyz_mol_ref, to_be_renumbered=smiles_mol_helper
        )
        # Check if the renumbering list is valid
        error_message = _check_renumbering_list(
            renum_list=renumbering_list, num_atoms=xyz_mol.GetNumAtoms()
        )
        if error_message is not None:
            return None, error_message

        smiles_mol = Chem.RenumberAtoms(smiles_mol, renumbering_list)

    # Clean up (after Hueckel calculation)
    clean_up(to_be_removed=["nul", "run.out"])

    return smiles_mol, _errmsg


def get_atom_bond_mapping_dicts(mol: Chem.rdchem.Mol) -> Tuple[Dict[int, int], Dict[int, int], str]:
    """Get index mapping dictionaries for atoms and bonds to map between two atom and bond orders
    that emerge when the SMILES string is canonicalized.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        An RDKit molecule object.

    Returns
    -------
    Tuple[Dict[int, int], Dict[int, int], str]
        A tuple containing:

        * A dictionary mapping from the canonical atom indices (keys) to the original atom
          indices (values).
        * A dictionary mapping from the canonical bond indices (keys) to the original bond
          indices (values).
        * The canonical SMILES string of the molecule (without hydrogen atoms).

    Notes
    -----
    When reading in a SMILES string with explicit hydrogen atoms with ``sanitize=False`` (followed
    by ``Chem.SanitizeMol()``), the atom order is different from when reading in the SMILES string
    with ``sanitize=True`` followed by ``Chem.AddHs()``. This becomes a problem when external
    programs read SMILES strings with hydrogen atoms without setting ``sanitize=False``.

    This means:

    * When an RDKit mol object generated from a canonical SMILES string without hydrogen atoms is
      passed to this function, no change in atom or bond order will be observed.
    * When an RDKit mol object generated from a canonical SMILES string WITH hydrogen atoms is
      passed to this function, a change in atom or bond order will be observed, even though the
      initial SMILES string was canonical.

    Essentially, a mapping of the input mol object to a mol object generated from
    ``Chem.MolFromSmiles()`` (optionally followed by ``Chem.AddHs()``) is performed.
    """
    helper_mol = Chem.Mol(mol)

    # Generate canonical smiles
    canonical_smiles = Chem.CanonSmiles(Chem.MolToSmiles(helper_mol))

    # Generate canonical mol object
    canonical_mol = Chem.MolFromSmiles(canonical_smiles)

    if helper_mol.GetNumAtoms() != canonical_mol.GetNumAtoms():
        canonical_mol = Chem.AddHs(canonical_mol)

    # Atoms
    renumbering_list = _get_renumbering_list(template=helper_mol, to_be_renumbered=canonical_mol)
    mapping_dict_atoms = {j: i for i, j in enumerate(renumbering_list)}
    mapping_dict_atoms = dict(sorted(mapping_dict_atoms.items()))

    # Bonds
    mapping_dict_bonds = {}
    for bond in canonical_mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        b = helper_mol.GetBondBetweenAtoms(
            mapping_dict_atoms[begin_atom_idx], mapping_dict_atoms[end_atom_idx]
        )
        mapping_dict_bonds[bond.GetIdx()] = b.GetIdx()

    return mapping_dict_atoms, mapping_dict_bonds, canonical_smiles


def get_charge_from_mol_object(mol: Chem.rdchem.Mol) -> int:
    """Get the formal charge of an RDKit molecule object.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        An RDKit molecule object.

    Returns
    -------
    int
        The formal charge of the molecule.
    """
    return int(Chem.GetFormalCharge(mol))


def from_periodic_table(
    periodic_table: Dict[str, element], element_symbol: str
) -> Tuple[Dict[str, element], element]:
    """Retrieve element data from the periodic table or create a new entry if it doesn't exist.

    The data is retrieved from the ``mendeleev`` library.

    Parameters
    ----------
    periodic_table : Dict[str, element]
        A dictionary representing the periodic table with element symbols as keys and mendeleev
        ``element`` objects as values.
    element_symbol : str
        The symbol of the element to retrieve.

    Returns
    -------
    Tuple[Dict[str, element], element]
        A tuple containing the updated periodic table and the requested element data.
    """
    element_data = periodic_table.get(element_symbol, None)
    if element_data is None:
        periodic_table[element_symbol] = element(element_symbol)
        element_data = periodic_table.get(element_symbol)
    return periodic_table, element_data


def get_ring_classification(mol: Chem.rdchem.Mol, ring_indices: List[int], idx_type: str) -> str:
    """Classify a ring based on its aromaticity and atom types either based on atom or bond
    indices.

    Possible classifications are:

    * "aromatic_carbocycle"
    * "aromatic_heterocycle"
    * "nonaromatic_carbocycle"
    * "nonaromatic_heterocycle"

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        An RDKit molecule object.
    ring_indices : List[int]
        A list of indices representing the atoms or bonds in the ring.
    idx_type : str
        The type of indices used, either "atom" or "bond".

    Returns
    -------
    str
        A string representing the classification of the ring.
    """
    # Get atom or bond objects
    if idx_type == "atom":
        ring_obs = [mol.GetAtomWithIdx(idx) for idx in ring_indices]
    if idx_type == "bond":
        ring_obs = [mol.GetBondWithIdx(idx) for idx in ring_indices]

    # Analyze atoms or bonds
    aromaticity_info = []
    atom_type_info = []
    ring_info = []
    for obj in ring_obs:
        aromaticity_info.append(obj.GetIsAromatic())
        if idx_type == "atom":
            atom_type_info.append(obj.GetSymbol())
            ring_info.append(obj.IsInRing())
        if idx_type == "bond":
            atom_type_info.append(obj.GetBeginAtom().GetSymbol())
            atom_type_info.append(obj.GetEndAtom().GetSymbol())
            ring_info.append(obj.IsInRing())

    # Check if all atoms are in a ring (should always be true, but just to be sure)
    if not all(ring_info):
        return "None"

    # Make classification
    if all(aromaticity_info) and set(atom_type_info) == set("C"):
        return "aromatic_carbocycle"
    elif all(aromaticity_info) and set(atom_type_info) != set("C"):
        return "aromatic_heterocycle"
    elif not all(aromaticity_info) and set(atom_type_info) == set("C"):
        return "nonaromatic_carbocycle"
    elif not all(aromaticity_info) and set(atom_type_info) != set("C"):
        return "nonaromatic_heterocycle"
    else:
        return "None"  # This should never happen, but just to be sure


def get_molecular_formula(mol: Chem.rdchem.Mol) -> str:
    """Calculate the molecular formula of an RDKit molecule object.

    Only atoms within the molecule object are considered. No hydrogen atoms are added.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        An RDKit molecule object.

    Returns
    -------
    str
        The molecular formula of the molecule.
    """
    counter_dict: Dict[str, int] = defaultdict(int)
    for atom in mol.GetAtoms():
        counter_dict[atom.GetSymbol()] += 1

    counter_dict_new = dict(sorted(counter_dict.items()))
    formula = [f"{key}{value}" for key, value in counter_dict_new.items()]
    formula_str = "".join(formula)

    return formula_str
