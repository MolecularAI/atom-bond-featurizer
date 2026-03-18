"""Utility functions for input/output operations."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional, Tuple

from rdkit import Chem

from bonafide.utils.constants import EH_TO_KJ_MOL, ELEMENT_SYMBOLS, ENERGY_UNITS, KCAL_MOL_TO_KJ_MOL

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def read_mol_object(
    mol: Chem.rdchem.Mol,
) -> Tuple[Chem.rdchem.Mol, List[Chem.rdchem.Mol], Optional[str]]:
    """Process an RDKit molecule object for incorporation into a molecule vault.

    The conformer molecule-level properties are moved to properties of the processed molecule
    objects.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        The RDKit molecule object to be processed. It can contain one or more conformers.

    Returns
    -------
    Tuple[Chem.rdchem.Mol, List[Chem.rdchem.Mol]]
        A tuple containing:

        * The initial input RDKit molecule object.
        * A list of RDKit molecule objects, each containing one conformer of the input molecule.
        * An error message if the input molecule object is not valid, otherwise ``None``.
    """
    _errmsg = None

    # Ensure validity of RDKit mol object
    # Sanitization is not performed as this might modify the user mol object in an undesired way.
    # Instead, the user is responsible to ensure that the input mol object is valid.
    if type(mol) != Chem.rdchem.Mol:
        _errmsg = "the input value is not a valid RDKit mol object"
        return mol, [], _errmsg

    if mol.GetNumConformers() < 1:
        return mol, [Chem.Mol(mol)], _errmsg

    # Generate a separate mol object for each conformer
    mols = []
    props = []
    for idx, conf in enumerate(mol.GetConformers()):
        if conf.Is3D() is False:
            _errmsg = f"the conformer with index {idx} is 2D; only 3D conformers are supported"
            return mol, [], _errmsg

        mol_ = Chem.Mol(mol)
        mol_.RemoveAllConformers()
        mol_.AddConformer(conf=conf, assignId=True)
        mols.append(mol_)
        props.append(conf.GetPropsAsDict())

    # Move conformer properties to mol object level
    for prop, m in zip(props, mols):
        for prop_name, value in prop.items():
            dtype = type(value).__name__

            if "int" in dtype:
                m.SetIntProp(prop_name, int(value))
            elif "float" in dtype:
                m.SetDoubleProp(prop_name, float(value))
            elif dtype == "str":
                m.SetProp(prop_name, value)
            elif dtype == "bool":
                m.SetBoolProp(prop_name, value)

    return mol, mols, _errmsg


def read_smiles(smiles: str) -> Tuple[Optional[Chem.rdchem.Mol], Optional[str]]:
    """Read a SMILES string and return an RDKit molecule object and an error message
    (``None`` if no error).

    Initially, ``sanitize=False`` is set in ``Chem.MolFromSmiles()`` to preserve the hydrogen atoms
    if they are given in the SMILES string. If the molecule object is successfully created, it is
    tried to be sanitized.

    Parameters
    ----------
    smiles : str
        The SMILES string of a molecule.

    Returns
    -------
    Tuple[Optional[Chem.rdchem.Mol], Optional[str]]
        A tuple containing:

        * An RDKit molecule object if the SMILES string could be parsed, otherwise ``None``.
        * An error message if the SMILES string could not be parsed or sanitized, otherwise
          ``None``.
    """
    _errmsg = None
    mol = None

    # Try to generate RDKit mol object without sanitization first
    try:
        mol = Chem.MolFromSmiles(SMILES=smiles, sanitize=False)
    except Exception as e:
        _errmsg = f"Generation of RDKit mol object failed for SMILES string '{smiles}': {e}."
        return mol, _errmsg

    # Try to sanitize the mol object
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            _errmsg = (
                f"Sanitization of the RDKit mol object generated from SMILES string "
                f"'{smiles}' failed: {e}."
            )
    else:
        _errmsg = (
            f"Generation of RDKit mol object failed for SMILES string '{smiles}' as it "
            "resulted in None."
        )

    return mol, _errmsg


def read_smarts(smarts: str) -> Tuple[Optional[Chem.rdchem.Mol], Optional[str]]:
    """Read a SMARTS pattern and return an RDKit molecule object and an error message
    (``None`` if no error).

    Parameters
    ----------
    smarts : str
        The SMARTS pattern.

    Returns
    -------
    Tuple[Optional[Chem.rdchem.Mol], Optional[str]]
        A tuple containing:

        * An RDKit molecule object if the SMARTS pattern could be parsed, otherwise ``None``.
        * An error message if the SMARTS pattern could not be parsed, otherwise ``None``.
    """
    _errmsg = None
    mol = None

    # Check if SMARTS string is empty
    smarts = smarts.strip()
    if smarts == "":
        _errmsg = "the SMARTS string must not be empty"
        return mol, _errmsg

    # Try to generate RDKit mol object
    try:
        mol = Chem.MolFromSmarts(SMARTS=smarts)
    except Exception as e:
        _errmsg = f"generation of RDKit mol object failed for SMARTS string '{smarts}': {e}"
        return mol, _errmsg

    # Check if mol is None
    if mol is None:
        _errmsg = (
            f"generation of RDKit mol object failed for SMARTS string '{smarts}' as it "
            "resulted in None"
        )

    return mol, _errmsg


def _validate_xyz(
    file_lines: List[str], number_of_atoms: int
) -> Tuple[List[str], List[str], Optional[str]]:
    """Validate the individual lines of an XYZ file with one or more conformers.

    The following points are ensured:

    * The first line of each structure block contains only a valid integer specifying the number of
      atoms in the block.
    * The number of atoms specified in the first line of each block matches the number of atoms
      specified in the first line of the first block.
    * Each atom line contains exactly one valid element symbol and three valid cartesian
      coordinates (x, y, z) that can be converted to floats.
    * The number of atom lines in each block matches the number of atoms specified in the first
      line of the file.
    * The elements in each block are identical and in the same order as found in the first
      structure block.

    Please note: These checks are not exhaustive and beyond them the user is responsible to ensure
    that the individual structure blocks represent conformers of the same molecule.

    Parameters
    ----------
    file_lines : List[str]
        The individual lines of the XYZ file.
    number_of_atoms : int
        The number of atoms in the molecule as defined by the first line of the XYZ file.

    Returns
    -------
    Tuple[List[str], List[str], Optional[str]]
        A tuple containing:

        * A list of the comment lines of each conformer block.
        * A list of strings, each string representing one conformer's atom lines.
        * An error message if the file lines are not valid, otherwise ``None``.
    """
    _errmsg = None

    # Get individual atom blocks and check if they are valid
    comment_line_in_blocks: List[str] = []
    elements = None
    atom_lines_in_blocks: List[str] = []

    for block_idx, start_line_idx in enumerate(range(0, len(file_lines), number_of_atoms + 2)):
        # Check if first line in block is a valid number of atoms line
        try:
            n_atoms_in_block = int(file_lines[start_line_idx])
        except Exception as e:
            _errmsg = (
                f"the first line of XYZ block with index {block_idx} is not a single valid "
                f"integer specifying the number of atoms in the molecule: {e}."
            )
            return comment_line_in_blocks, atom_lines_in_blocks, _errmsg

        # Check if number of atoms in block matches number of atoms in first block
        if n_atoms_in_block != number_of_atoms:
            _errmsg = (
                f"the number of atoms specified in the first line of XYZ block with index "
                f"{block_idx} ({n_atoms_in_block}) does not match the number of atoms "
                f"specified in the XYZ block with index 0 ({number_of_atoms})"
            )
            return comment_line_in_blocks, atom_lines_in_blocks, _errmsg

        # Add comment line of block to list of all comment lines
        comment_line_in_blocks.append(file_lines[start_line_idx + 1])

        # Loop over individual atom lines in block (skip first two lines)
        atom_lines_in_block = []
        elements_in_block = []
        for line in file_lines[start_line_idx + 2 : start_line_idx + 2 + number_of_atoms]:
            splitted = line.split()

            # Check file format (element symbol + 3 coordinates per line)
            if len(splitted) != 4:
                _errmsg = (
                    f"a line in xzy block with index {block_idx} does not contain "
                    "exactly one element symbol and three cartesian coordinates (x, y, z)"
                )
                return comment_line_in_blocks, atom_lines_in_blocks, _errmsg

            # Check if element symbol is valid
            if splitted[0] not in ELEMENT_SYMBOLS:
                _errmsg = (
                    f"element symbol '{splitted[0]}' in xzy block with index "
                    f"{block_idx} is not a valid element symbol"
                )
                return comment_line_in_blocks, atom_lines_in_blocks, _errmsg

            # Check if coordinates are valid floats
            try:
                for coord in splitted[1:]:
                    float(coord)
            except Exception as e:
                _errmsg = (
                    f"one of the coordinates in xzy block with index {block_idx} is not a "
                    f"valid float: {e}."
                )
                return comment_line_in_blocks, atom_lines_in_blocks, _errmsg

            # Add atom line and element
            atom_lines_in_block.append(line)
            elements_in_block.append(splitted[0])

        # Check if number of atom lines matches number of atoms
        if len(atom_lines_in_block) != number_of_atoms:
            _errmsg = (
                f"the number of atom lines in XYZ block with index {block_idx} "
                f"({len(atom_lines_in_block)}) does not match the number of atoms of the molecule "
                f"specified in the first line of the file ({number_of_atoms})"
            )
            return comment_line_in_blocks, atom_lines_in_blocks, _errmsg

        # Check if elements are consistent for all conformers
        if elements is not None:
            if elements != elements_in_block:
                _errmsg = (
                    f"the elements in the XYZ block of conformer with index {block_idx} are not "
                    "identical and/or in the same order as found in the conformer with index 0"
                )
                return comment_line_in_blocks, atom_lines_in_blocks, _errmsg
        else:
            elements = elements_in_block

        # Add atom lines of block to list of all blocks
        atom_lines_in_blocks.append("".join(atom_lines_in_block))

    return comment_line_in_blocks, atom_lines_in_blocks, _errmsg


def read_xyz_file(file_path: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Read an XYZ file with one or more conformers and validate its content.

    The first line of each conformer block contains the number of atoms, the second line is a
    comment line, and the subsequent lines contain the atom symbols and their cartesian coordinates
    (in Angstrom). The individual conformers cannot be separated by empty lines. The file content
    is validated (see ``_validate_xyz()`` for details).

    Parameters
    ----------
    file_path : str
        The path to the XYZ file.

    Returns
    -------
    Tuple[Optional[List[str]], Optional[str]]
        A tuple containing:

        * A list of strings, each representing one conformer's XYZ block.
        * An error message if the file could not be read or is not valid, otherwise ``None``.
    """
    _errmsg = None
    xyz_blocks = None

    # Ensure correct file extension (just for double-checking)
    if not file_path.endswith(".xyz"):
        _errmsg = "the file does not have the expected .xyz file extension"
        return xyz_blocks, _errmsg

    # Try to open the file
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        _errmsg = f"opening of the file failed: {e}"
        return xyz_blocks, _errmsg

    # Strip away empty lines at the end of the file
    while len(lines) != 0 and lines[-1].strip() == "":
        lines.pop(-1)

    # Check if file is empty
    if len(lines) == 0:
        _errmsg = "the file is empty or only contains empty lines"
        return xyz_blocks, _errmsg

    # Try to get the number of atoms of the first conformer
    try:
        n_atoms = int(lines[0])
    except Exception:
        _errmsg = (
            "the first line of the file is not a single valid integer defining the number "
            "of atoms in the molecule"
        )
        return xyz_blocks, _errmsg

    # Check if at least as many lines as atoms + 2 are present
    if len(lines) < n_atoms + 2:
        _errmsg = (
            f"the file contains fewer non-empty lines ({len(lines)}) than required for a single "
            f"structure block with {n_atoms} atoms"
        )
        return xyz_blocks, _errmsg

    # Validate the input file
    comment_line_in_blocks, atom_lines_in_blocks, _errmsg = _validate_xyz(
        file_lines=lines, number_of_atoms=n_atoms
    )
    if _errmsg is not None:
        return xyz_blocks, _errmsg

    # Format XYZ blocks
    try:
        xyz_blocks = []
        for comment, atom_lines in zip(comment_line_in_blocks, atom_lines_in_blocks):
            xyz_block = f"{n_atoms}\n{comment}{atom_lines}"
            xyz_blocks.append(xyz_block)
    except Exception as e:
        _errmsg = (
            "the file was successfully read and validated, but the "
            f"final formatting of the individual XYZ blocks failed: {e}"
        )

    return xyz_blocks, _errmsg


def _validate_sdf(sdf_mols: List[Optional[Chem.rdchem.Mol]]) -> Optional[str]:
    """Validate the individual RDKit molecule objects generated from an SD file with one or more
    conformers.

    The following points are ensured:

    * All conformers could be successfully converted to RDKit molecule objects that are not
      ``None``.
    * All elements in the conformers represent valid element symbols.
    * All conformers represent the same molecule (checked by comparing their SMILES and InChIKey
      string as well as their chemical element symbols).
    * All conformers possess 3D coordinates.

    Parameters
    ----------
    sdf_mols : List[Optional[Chem.rdchem.Mol]]
        A list of RDKit molecule objects generated from the SD file (see the ``read_sd_file()``
        function). ``None`` can be present in the list if individual conformers could not be
        parsed.

    Returns
    -------
    Optional[str]
        An error message if the molecule objects are not valid, otherwise ``None``.
    """
    _errmsg = None

    _smiles = None
    _inchikey = None
    _elements = None

    for idx, mol in enumerate(sdf_mols):
        # Check if mol is None
        if mol is None:
            _errmsg = (
                f"generation of RDKit mol object from SDF block failed for conformer "
                f"with index {idx} as it resulted in None"
            )
            return _errmsg

        # Validate elements
        els = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for el in els:
            if el not in ELEMENT_SYMBOLS:
                _errmsg = (
                    f"element symbol '{el}' in SDF block with index {idx} is not a valid "
                    "element symbol"
                )
                return _errmsg

        # Compare conformers by SMILES, InChIKey, and elements
        smi = Chem.MolToSmiles(mol)
        ikey = Chem.MolToInchiKey(mol)

        if _smiles is None:
            _smiles = smi
            _inchikey = ikey
            _elements = els
        else:
            # SMILES
            if smi != _smiles:
                _errmsg = (
                    f"the generated SMILES string of the conformer with index {idx} "
                    f"('{smi}') does not match the SMILES string of the conformer with index 0 "
                    f"('{_smiles}')"
                )
                return _errmsg

            # InChIKey
            if ikey != _inchikey:
                _errmsg = (
                    f"the generated InChIKey of the conformer with index {idx} "
                    f"('{ikey}') does not match the InChIKey of the conformer with index 0 "
                    f"('{_inchikey}')"
                )
                return _errmsg

            # Elements
            if els != _elements:
                _errmsg = (
                    f"the elements of the conformer with index {idx} are not identical "
                    "and/or in the same order as found in the conformer with index 0"
                )
                return _errmsg

        # Ensure that conformer is 3D
        if mol.GetConformer().Is3D() is False:
            _errmsg = (
                f"the conformer with index {idx} is not 3D. Only SD files containing 3D "
                "information are supported"
            )
            return _errmsg

    return _errmsg


def read_sd_file(
    file_path: str,
) -> Tuple[Optional[List[Optional[Chem.rdchem.Mol]]], Optional[str]]:
    """Read an SD file with one or more conformers.

    The file must comply with the SD file format (see
    https://en.wikipedia.org/wiki/Chemical_table_file, last accessed on 23.09.2025).

    Parameters
    ----------
    file_path : str
        Path to the SD file.

    Returns
    -------
    Tuple[Optional[List[Optional[Chem.rdchem.Mol]]], Optional[str]]
        A tuple containing:

        * A list of RDKit molecule objects if the file could be read , otherwise ``None``. The
          mol objects can also be ``None`` if individual conformers could not be parsed.
        * An error message if the file could not be read or is not valid, otherwise ``None``.
    """
    sdf_mols = None
    _errmsg = None

    # Ensure correct file extension (just for double-checking)
    if not file_path.endswith(".sdf"):
        _errmsg = "the file does not have the expected .sdf file extension"
        return sdf_mols, _errmsg

    # Try to open the file
    try:
        suppl = Chem.SDMolSupplier(file_path, sanitize=False)
    except Exception as e:
        _errmsg = f"opening of the file failed: {e}"
        return sdf_mols, _errmsg

    # Get mol objects
    sdf_mols = [mol for mol in suppl]

    # Validate the mol objects
    _errmsg = _validate_sdf(sdf_mols)
    if _errmsg is not None:
        _errmsg = f"validation of the file failed: {_errmsg}"

    return sdf_mols, _errmsg


def write_sd_file(mol: Chem.rdchem.Mol, file_path: str) -> None:
    """Write an SD file from an RDKit mol object.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        An RDKit molecule object.
    file_path : str
        The path to the file the data is written to.

    Returns
    -------
    None
    """
    with Chem.SDWriter(file_path) as writer:
        writer.write(mol)


def extract_energy_from_string(
    line: str,
) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]:
    """Read the energy and its unit from a string and convert it to kJ/mol.

    Supported energy units are: kcal/mol, kJ/mol, and Eh (Hartree).

    Parameters
    ----------
    line : str
        A string containing the energy value and its unit.

    Returns
    -------
    Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]
        A tuple containing:

        * The energy value as submitted if found (or ``None`` if no valid energy is found)
        * The unit as submitted if found (or ``None`` if no valid unit is found)
        * The energy value converted to kJ/mol (or ``None`` if no valid energy is found)
        * An error message (``None`` if no error occurred).
    """
    # Energy conversions
    _conversions = {
        "kj_mol": 1,
        "kcal_mol": KCAL_MOL_TO_KJ_MOL,
        "eh": EH_TO_KJ_MOL,
    }

    _line = line
    line = str(line).lower()

    energy_as_submitted = None
    unit_as_submitted = None
    energy = None
    _errmsg = None

    # Search for energy value and unit in string
    for base_unit, symbol_list in ENERGY_UNITS.items():
        for symbol in symbol_list:
            pattern = (
                rf"([-+]?(?:\d{{1,3}}(?:,\d{{3}})*|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*{symbol}"
            )
            matches = re.findall(pattern, line)

            if len(matches) > 0:
                energy_as_submitted = matches[0]
                energy_as_submitted = float(energy_as_submitted.replace(",", ""))
                unit_as_submitted = base_unit
                break

    # Check if data was found and convert to kJ/mol
    if energy_as_submitted is not None and unit_as_submitted is not None:
        energy_as_submitted = float(energy_as_submitted)
        energy = float(energy_as_submitted * _conversions[unit_as_submitted])
    else:
        _errmsg = (
            f"no valid energy value with a supported unit (kJ/mol, kcal/mol, Eh) could be "
            f"extracted from the string '{_line}'"
        )

    return energy_as_submitted, unit_as_submitted, energy, _errmsg


def write_xyz_file_from_coordinates_array(
    elements: NDArray[np.str_],
    coordinates: NDArray[np.float64],
    file_path: str,
) -> None:
    """Write a list of elements and their coordinates to an XYZ file.

    Parameters
    ----------
    elements : NDArray[np.str\_]
        The element symbols of the molecule.
    coordinates : NDArray[np.float64]
        The cartesian coordinates of the structure.
    file_path : str
        The path to the output XYZ file.

    Returns
    -------
    None
    """
    # Format file content
    xyz = f"{len(elements)}\n\n"
    for el, coord in zip(elements, coordinates):
        xyz += f"{el}    {coord[0]}    {coord[1]}    {coord[2]}\n"

    # Write file
    with open(file_path, "w") as f:
        f.write(xyz)
