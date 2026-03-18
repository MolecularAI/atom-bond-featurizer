"""Data class for storing all the information on a molecule and its conformers."""

from __future__ import annotations

import io
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import ipywidgets
import numpy as np
import py3Dmol
from rdkit import Chem
from rdkit.Chem import Draw

from bonafide.utils.constants import UNDESIRED_ATOM_BOND_PROPERTIES, R
from bonafide.utils.helper_functions import get_function_or_method_name, standardize_string
from bonafide.utils.helper_functions_chemistry import get_charge_from_mol_object
from bonafide.utils.io_ import extract_energy_from_string, read_smiles
from bonafide.utils.string_formatting import _make_bold_end, _make_bold_start

if TYPE_CHECKING:
    import traitlets.utils.bunch.Bunch
    from numpy.typing import NDArray
    from PIL import PngImagePlugin


@dataclass
class MolVault:
    """A dataclass for storing all information on the molecule under consideration including its
    conformers.

    The calculated atom and bond features are stored as atom and bond properties, respectively, of
    the RDKit molecule objects in the ``mol_objects`` attribute. Additionally, the calculated
    features are cached in respective dictionaries.

    Attributes
    ----------
    input_type : str
        The type of input data, either "smiles", "xyz", "sdf", or "mol_object".
    mol_inputs : Union[List[str], Tuple[Chem.rdchem.Mol, List[Chem.rdchem.Mol]]]
        The formatted molecule input data to initialize the molecule vault. The data type depends
        on the input type:

        * input_type="smiles": A list of length 1 containing the SMILES string of the molecule.
        * input_type="xyz": A list of XYZ blocks as strings, one for each conformer.
        * input_type="sdf": A list of RDKit molecule objects, one for each conformer.
        * input_type="mol_object": A tuple of length 2, where the first entry the input RDKit
          molecule object and the second entry is a list of RDKit molecule objects, one for each
          conformer.
    namespace : str
        The namespace of the provided input as defined by the user.

    Returns
    -------
    None
    """

    mol_inputs: Union[List[str], Tuple[Chem.rdchem.Mol, List[Chem.rdchem.Mol]]]
    namespace: str
    input_type: str

    def __post_init__(self) -> None:
        """Post-initialization of additional attributes.

        Attributes
        ----------
        _input_energies_n : List[Tuple[Optional[float], Optional[str]]]
            The energy of each conformer from the input and the associated unit as provided by the
            user.
        _input_energies_n_minus1 : List[Tuple[Optional[float], Optional[str]]]
            The energy of the one-electron-oxidized molecule for each conformer from the input and
            the associated unit as provided by the user.
        _input_energies_n_plus1 : List[Tuple[Optional[float], Optional[str]]]
            The energy of the one-electron-reduced molecule for each conformer from the input and
            the associated unit as provided by the user.
        _input_mol_objects : Union[Chem.rdchem.Mol, List[Chem.rdchem.Mol]]
            The RDKit molecule object(s) from the original user input.
        atom_feature_cache_n : List[Dict[str, Dict[int, Optional[Union[str, bool, int, float]]]]]
            The cache of atom features for each conformer. The individual list entries are
            dictionaries with the feature names as keys and dictionaries mapping atom indices to
            feature values as values.
        atom_feature_cache_n_minus1 : List[Dict[str, Dict[int, Optional[Union[str, bool, int, float]]]]]
            The cache of atom features for the one-electron-oxidized molecule for each conformer.
            The individual list entries are dictionaries with the feature names as keys and
            dictionaries mapping atom indices to feature values as values.
        atom_feature_cache_n_plus1 : List[Dict[str, Dict[int, Optional[Union[str, bool, int, float]]]]]
            The cache of atom features for the one-electron-reduced molecule for each conformer.
            The individual list entries are dictionaries with the feature names as keys and
            dictionaries mapping atom indices to feature values as values.
        boltzmann_weights : Tuple[Optional[Union[int, float]], Optional[List[Optional[float]]]]
            The first element in the tuple is the temperature at which the Boltzmann weights were
            computed. The second entry represents the Boltzmann weight for each conformer,
            computed from ``energies_n``.
        bond_feature_cache : List[Dict[str, Dict[int, Optional[Union[str, bool, int, float]]]]]
            The cache of bond features for each conformer. The individual list entries are
            dictionaries with the feature names as keys and dictionaries mapping bond indices to
            feature values as values.
        bonds_determined : bool
            Indicates if bond information for the molecule is available or has been determined.
        charge : Optional[int]
            The total charge of the molecule.
        conformer_names : List[str]
            The names of each conformer, generated using the input name as given by the user and
            the conformer index.
        dimensionality : str
            The dimensionality of the molecule in the molecule vault ("2D" or "3D").
        electronic_struc_types_n : List[Optional[str]]
            The file extension of the electronic structure files for each conformer.
        electronic_struc_types_n_minus1 : List[Optional[str]]
            The file extension of the electronic structure files for the one-electron-oxidized
            molecule for each conformer.
        electronic_struc_types_n_plus1 : List[Optional[str]]
            The file extensions of the electronic structure files for the one-electron-reduced
            molecule for each conformer.
        electronic_strucs_n : List[Optional[str]]
            The path to the electronic structure files for each conformer.
        electronic_strucs_n_minus1 : List[Optional[str]]
            The path to the electronic structure files for the one-electron-oxidized molecule for
            each conformer.
        electronic_strucs_n_plus1 : List[Optional[str]]
            The path to the electronic structure files for the one-electron-reduced molecule for
            each conformer.
        elements : NDArray[np.str\_]
            The element symbols of the molecule.
        energies_n : List[Tuple[Optional[float], str]]
            The energy of each conformer and the unit (kJ/mol) as a string.
        energies_n_minus1 : List[Tuple[Optional[float], str]]
            The energy for the one-electron-oxidized molecule of each conformer and the unit
            (kJ/mol) as a string.
        energies_n_minus1_read : bool
            Indicates if the energies of the one-electron-oxidized conformers have been read.
        energies_n_plus1 : List[Tuple[Optional[float], str]]
            The energy for the one-electron-reduced molecule of each conformer and the unit
            (kJ/mol) as a string.
        energies_n_plus1_read : bool
            Indicates if the energies of the one-electron-reduced conformers have been read.
        energies_n_read : bool
            Indicates if the energies of the conformers have been read.
        global_feature_cache : List[Dict[str, Optional[Union[str, bool, int, float]]]]
            The cache of global features for each conformer. The individual list entries are
            dictionaries with the feature names as keys and feature values as values.
        is_valid : List[bool]
            Indicates if each conformer is valid (``True``) or not (``False``).
        mol_objects : List[Chem.rdchem.Mol]
            The RDKit molecule object for each conformer. They are used to store the calculated
            atom and bond features as properties of the individual atoms or bonds.
        multiplicity : Optional[int]
            The spin multiplicity of the molecule.
        size : int
            The number of conformers in the molecule vault. If a SMILES string is read, this is set
            to 0.
        smiles : Optional[str]
            The SMILES string of the molecule.

        Returns
        -------
        None
        """
        self.mol_objects: List[Chem.rdchem.Mol] = []
        self.conformer_names: List[str] = []
        self.dimensionality: Optional[str] = None
        self.size: Optional[int] = None
        self.elements: Optional[NDArray[np.str_]] = None
        self.charge: Optional[int] = None
        self.multiplicity: Optional[int] = None
        self.is_valid: List[bool] = []
        self.energies_n: List[Tuple[Optional[float], str]] = []
        self.energies_n_minus1: List[Tuple[Optional[float], str]] = []
        self.energies_n_plus1: List[Tuple[Optional[float], str]] = []
        self.energies_n_read: bool = False
        self.energies_n_minus1_read: bool = False
        self.energies_n_plus1_read: bool = False
        self.boltzmann_weights: Union[
            Tuple[Optional[Union[int, float]], Optional[List[Optional[float]]]], Tuple[()]
        ] = ()
        self.electronic_strucs_n: List[Optional[str]] = []
        self.electronic_strucs_n_minus1: List[Optional[str]] = []
        self.electronic_strucs_n_plus1: List[Optional[str]] = []
        self.electronic_struc_types_n: List[Optional[str]] = []
        self.electronic_struc_types_n_minus1: List[Optional[str]] = []
        self.electronic_struc_types_n_plus1: List[Optional[str]] = []
        self.smiles: Optional[str] = None
        self.bonds_determined: bool = False
        self.atom_feature_cache_n: List[
            Dict[str, Dict[int, Optional[Union[str, bool, int, float]]]]
        ] = []
        self.atom_feature_cache_n_minus1: List[
            Dict[str, Dict[int, Optional[Union[str, bool, int, float]]]]
        ] = []
        self.atom_feature_cache_n_plus1: List[
            Dict[str, Dict[int, Optional[Union[str, bool, int, float]]]]
        ] = []
        self.bond_feature_cache: List[
            Dict[str, Dict[int, Optional[Union[str, bool, int, float]]]]
        ] = []
        self.global_feature_cache: List[Dict[str, Optional[Union[str, bool, int, float]]]] = []
        self._input_mol_objects: Union[Chem.rdchem.Mol, List[Chem.rdchem.Mol]] = []
        self._input_energies_n: List[Tuple[Optional[float], Optional[str]]] = []
        self._input_energies_n_minus1: List[Tuple[Optional[float], Optional[str]]] = []
        self._input_energies_n_plus1: List[Tuple[Optional[float], Optional[str]]] = []

    def __repr__(self) -> str:
        """A custom string representation of the ``MolVault`` object.

        Returns
        -------
        str
            The formatted string representation of the ``MolVault`` object.
        """
        repr_str = ""
        for att, val in self.__dict__.items():
            if att == "elements":
                val = [str(el) for el in val.tolist()]
            repr_str += f"{_make_bold_start}{att}:{_make_bold_end} "
            repr_str += str(val) + "\n"
        return repr_str

    def initialize_mol(self) -> None:
        """Initialize the molecule from the input data, either from XYZ or SDF blocks, from a
        SMILES string, or from RDKit molecule objects. This includes the initialization of all
        conformers (in case of XYZ, SDF, or RDKit molecule object input).

        Returns
        -------
        None
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Pre-process mol_inputs attribute for mol_object input
        if self.input_type == "mol_object":
            _init_mol = self.mol_inputs[0]
            self.mol_inputs = self.mol_inputs[1]  # type: ignore[assignment]

        self.size = len(self.mol_inputs)

        for idx, mol_entity in enumerate(self.mol_inputs):
            mol = None

            # From XYZ file
            if self.input_type == "xyz":
                try:
                    mol = Chem.MolFromXYZBlock(mol_entity)
                except Exception as e:
                    _errmsg = (
                        "Generation of RDKit mol object from XYZ block failed for conformer "
                        f"with index {idx}: {e}."
                    )
                    self.is_valid.append(False)
                    logging.error(f"'{self.namespace}' | {_loc}()\n{_errmsg}")
                    raise ValueError(f"{_loc}(): {_errmsg}")
                else:
                    if mol is None:
                        _errmsg = (
                            "Generation of RDKit mol object from XYZ block resulted in None "
                            f"for conformer with index {idx}."
                        )
                        self.is_valid.append(False)
                        logging.error(f"'{self.namespace}' | {_loc}()\n{_errmsg}")
                        raise ValueError(f"{_loc}(): {_errmsg}")
                    else:
                        logging.info(
                            f"'{self.namespace}' | {_loc}()\nGeneration of RDKit mol object from "
                            f"XYZ block successful for conformer with index {idx}."
                        )

                self.dimensionality = "3D"

            # From SD file
            if self.input_type == "sdf":
                mol = mol_entity

                # Overwrite the mol object in the mol_inputs attribute to contain the actual
                # SDF string input
                _sdf_string_ = io.StringIO()
                with Chem.SDWriter(_sdf_string_) as w:
                    w.write(mol)

                _sdf_string = _sdf_string_.getvalue()
                _sdf_lines = _sdf_string.splitlines()
                _sdf_lines[2] = f"RDKit-processed user input for conformer with index {idx}."
                _sdf_string = "\n".join(_sdf_lines)

                assert isinstance(self.mol_inputs, list)  # for type checker
                self.mol_inputs[idx] = _sdf_string

                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    _errmsg = (
                        f"Sanitization of the RDKit mol object generated from the SDF block "
                        f"of conformer with index {idx} failed: {e}."
                    )
                    self.is_valid.append(False)
                    logging.error(f"'{self.namespace}' | {_loc}()\n{_errmsg}")
                    raise ValueError(f"{_loc}(): {_errmsg}")

                logging.info(
                    f"'{self.namespace}' | {_loc}()\nGeneration of RDKit mol object from SDF block "
                    f"successful for conformer with index {idx}."
                )

                self.dimensionality = "3D"
                self.bonds_determined = True

            # From SMILES
            if self.input_type == "smiles":
                self.smiles = self.mol_inputs[0]
                mol, error_message = read_smiles(self.smiles)
                if error_message is not None:
                    self.is_valid.append(False)
                    logging.error(f"'{self.namespace}' | {_loc}()\n{error_message}")
                    raise ValueError(f"{_loc}(): {error_message}")
                else:
                    logging.info(
                        f"'{self.namespace}' | {_loc}()\nGeneration of RDKit mol object from "
                        "SMILES string successful."
                    )

                    self.dimensionality = "2D"
                    self.size = 0
                    self.bonds_determined = True
                    self.energies_n.append((None, "kj_mol"))
                    self.energies_n_minus1.append((None, "kj_mol"))
                    self.energies_n_plus1.append((None, "kj_mol"))
                    self.electronic_strucs_n.append(None)
                    self.electronic_strucs_n_minus1.append(None)
                    self.electronic_strucs_n_plus1.append(None)
                    self.electronic_struc_types_n.append(None)
                    self.electronic_struc_types_n_minus1.append(None)
                    self.electronic_struc_types_n_plus1.append(None)
                    self.boltzmann_weights = (None, [None])
                    self.charge = get_charge_from_mol_object(mol=mol)

            # From RDKit molecule object
            if self.input_type == "mol_object":
                mol = mol_entity

                if mol.GetNumConformers() > 0:
                    self.dimensionality = "3D"
                else:
                    self.dimensionality = "2D"
                    self.size = 0
                    self.energies_n.append((None, "kj_mol"))
                    self.energies_n_minus1.append((None, "kj_mol"))
                    self.energies_n_plus1.append((None, "kj_mol"))
                    self.electronic_strucs_n.append(None)
                    self.electronic_strucs_n_minus1.append(None)
                    self.electronic_strucs_n_plus1.append(None)
                    self.electronic_struc_types_n.append(None)
                    self.electronic_struc_types_n_minus1.append(None)
                    self.electronic_struc_types_n_plus1.append(None)
                    self.boltzmann_weights = (None, [None])
                    self.charge = get_charge_from_mol_object(mol=mol)

                if mol.GetNumBonds() > 0:
                    self.bonds_determined = True

                self.smiles = Chem.MolToSmiles(mol, canonical=False)

                logging.info(
                    f"'{self.namespace}' | {_loc}()\nGeneration of RDKit mol object from RDKit "
                    f"mol object successful for conformer with index {idx}."
                )

            # Save data
            self.mol_objects.append(mol)
            self.conformer_names.append(f"{self.namespace}__conf-{idx}")
            self._input_mol_objects.append(Chem.Mol(mol))
            self.atom_feature_cache_n.append({})
            self.atom_feature_cache_n_minus1.append({})
            self.atom_feature_cache_n_plus1.append({})
            self.bond_feature_cache.append({})
            self.global_feature_cache.append({})

            if mol is not None:
                self.is_valid.append(True)
            else:
                self.is_valid.append(False)

        # Post-processing for mol_object and smiles input
        if self.input_type == "mol_object":
            self.mol_inputs = [_init_mol]
            self._input_mol_objects = _init_mol
        if self.input_type == "smiles":
            self._input_mol_objects = self._input_mol_objects[0]

        # Clean properties after reading in the molecule (reading sdf sets properties)
        self.clean_properties()

    def get_elements(self) -> None:
        """Get the elements of the molecule.

        The zeroth conformer is used to extract the elements.

        Returns
        -------
        None
        """
        elements = []
        for atom in self.mol_objects[0].GetAtoms():
            elements.append(atom.GetSymbol())
        self.elements = np.array(elements)

    def read_mol_energies(self) -> None:
        """Read the energies of the conformers from the input data, either from XYZ or SDF data.

        Returns
        -------
        None
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        for idx, mol in enumerate(self.mol_objects):
            energy_as_submitted = None
            unit_as_submitted = None
            energy = None

            # From XYZ file
            if self.input_type == "xyz":
                try:
                    energy_as_submitted, unit_as_submitted, energy, error_message = (
                        self._extract_energy_from_xyz_block(xyz_block=self.mol_inputs[idx])  # type: ignore[arg-type]
                    )
                except Exception as e:
                    _errmsg = (
                        "Extraction of energy from XYZ block failed for conformer with "
                        f"index {idx}: {e}."
                    )
                    logging.error(f"'{self.namespace}' | {_loc}()\n{_errmsg}")
                    raise RuntimeError(f"{_loc}(): {_errmsg}")

                else:
                    if error_message is not None:
                        _errmsg = (
                            "Extraction of energy from XYZ block resulted in None for "
                            f"conformer with index {idx}: {error_message}."
                        )
                        logging.error(f"'{self.namespace}' | {_loc}()\n{_errmsg}")
                        raise ValueError(f"{_loc}(): {_errmsg}")
                    else:
                        logging.info(
                            f"'{self.namespace}' | {_loc}()\nExtraction of energy from XYZ block "
                            f"successful for conformer with index {idx}."
                        )

            # From SD file or RDKit molecule object
            if self.input_type in ["sdf", "mol_object"]:
                if self.input_type == "sdf":
                    _id_str = "SDF"
                else:
                    _id_str = "RDKit"

                try:
                    energy_as_submitted, unit_as_submitted, energy, error_message = (
                        self._extract_energy_from_mol_object(mol=mol)
                    )
                except Exception as e:
                    _errmsg = (
                        f"Extraction of energy from {_id_str} mol failed for conformer with "
                        f"index {idx}: {e}."
                    )
                    logging.error(f"'{self.namespace}' | {_loc}()\n{_errmsg}")
                    raise RuntimeError(f"{_loc}(): {_errmsg}")
                else:
                    if error_message is not None:
                        _errmsg = (
                            f"Extraction of energy from {_id_str} mol resulted in None for "
                            f"conformer with index {idx}: {error_message}."
                        )
                        logging.error(f"'{self.namespace}' | {_loc}()\n{_errmsg}")
                        raise ValueError(f"{_loc}(): {_errmsg}")
                    else:
                        logging.info(
                            f"'{self.namespace}' | {_loc}()\nExtraction of energy from {_id_str} "
                            f"mol successful for conformer with index {idx}.",
                        )

            # Save data
            self._input_energies_n.append((energy_as_submitted, unit_as_submitted))
            self.energies_n.append((energy, "kj_mol"))
            if energy is None:
                self.is_valid[idx] = False

        self.energies_n_read = True

    def render_mol(
        self, idx_type: Optional[str], in_3D: bool, image_size: Tuple[int, int]
    ) -> Union[PngImagePlugin.PngImageFile, ipywidgets.VBox]:
        """Display the molecule in a Jupyter notebook, optionally with atom or bond indices added
        to the structure.

        Parameters
        ----------
        idx_type : Optional[str]
            The type of indices to add to the structure, either "atom", "bond", or ``None``.
        in_3D : bool
            Whether to display the molecule in 3D (``True``) or as a 2D depiction (``False``).
        image_size : Tuple[int, int]
            The size of the generated image in pixels as a 2-tuple.

        Returns
        -------
        Union[PngImagePlugin.PngImageFile, ipywidgets.VBox]
            A 2D or 3D depiction of the molecule, either as an image or an interactive 3D view.
        """
        if in_3D is True:
            # Get all MOL blocks of the conformers
            mol_blocks = []
            for mol in self.mol_objects:
                mol_block = Chem.MolToMolBlock(mol)
                mol_blocks.append(mol_block)

            # Generate the 3D interactive view
            return self._render_mol_3D(
                mol_blocks=mol_blocks, image_size=image_size, idx_type=idx_type
            )

        # 2D depiction
        helper = Chem.Mol(self.mol_objects[0])
        helper.RemoveAllConformers()

        image = Draw.rdMolDraw2D.MolDraw2DSVG(0, 0)
        options = image.drawOptions()
        options.annotationFontScale = 0.8
        options.setAnnotationColour((0 / 255, 0 / 255, 255 / 255, 140 / 255))

        # Add indices
        if idx_type == "atom":
            options.addAtomIndices = True
            options.addBondIndices = False
        elif idx_type == "bond":
            options.addAtomIndices = False
            options.addBondIndices = True
        else:
            options.addAtomIndices = False
            options.addBondIndices = False

        return Draw.MolToImage(helper, size=image_size, options=options, legend=self.namespace)

    def prune_ensemble_by_energy(
        self, energy_cutoff: Tuple[Union[int, float], str], _called_from: str
    ) -> None:
        """Remove conformers from the ensemble that have a relative energy above a certain cutoff
        value.

        Parameters
        ----------
        energy_cutoff : Tuple[Union[int, float], str]
            A 2-tuple containing the cutoff energy value as the first entry and the unit as the
            second.
        _called_from : str
            The name of the method from which this method was called. This is only used for
            logging purposes.
        """
        # Check if energies are available
        if self.energies_n_read is False:
            _errmsg = (
                "The molecule vault does not contain energy information on the individual "
                "conformers. Therefore, the conformer ensemble cannot be pruned. Read in energies "
                "through the input file (read_input() method) or calculate the energies from "
                "scratch (calculate_electronic_structure() method) before requesting pruning."
            )
            logging.error(f"'{self.namespace}' | {_called_from}()\n{_errmsg}")
            raise ValueError(f"{_called_from}(): {_errmsg}")

        # Check input type
        _inpt: Any
        _inpt = type(energy_cutoff)
        if _inpt != tuple:
            _errmsg = (
                "Invalid input to 'prune_by_energy': must be of type tuple "
                f"but obtained {_inpt.__name__}."
            )
            logging.error(f"'{self.namespace}' | {_called_from}()\n{_errmsg}")
            raise TypeError(f"{_called_from}(): {_errmsg}")

        if len(energy_cutoff) != 2:
            _errmsg = (
                "Invalid input to 'prune_by_energy': must be a 2-tuple "
                f"but obtained a tuple of length {len(energy_cutoff)}."
            )
            logging.error(f"'{self.namespace}' | {_called_from}()\n{_errmsg}")
            raise ValueError(f"{_called_from}(): {_errmsg}")

        cutoff_value = energy_cutoff[0]
        unit = energy_cutoff[1]

        _inpt = type(cutoff_value)
        if isinstance(cutoff_value, (int, float)) is False:
            _errmsg = (
                "Invalid input to 'prune_by_energy': the first entry of the 2-tuple must be of "
                f"type int or float but obtained {_inpt.__name__}."
            )
            logging.error(f"'{self.namespace}' | {_called_from}()\n{_errmsg}")
            raise TypeError(f"{_called_from}(): {_errmsg}")

        _inpt = type(unit)
        if isinstance(unit, str) is False:
            _errmsg = (
                "Invalid input to 'prune_by_energy': the second entry of the 2-tuple must be of "
                f"type str but obtained {_inpt.__name__}."
            )
            logging.error(f"'{self.namespace}' | {_called_from}()\n{_errmsg}")
            raise TypeError(f"{_called_from}(): {_errmsg}")

        # Handle energy input (check unit and convert to kJ/mol)
        _energy_value_pair = f"{cutoff_value} {unit}"
        try:
            _, _, cutoff_value_, error_message = extract_energy_from_string(line=_energy_value_pair)
        except Exception as e:
            _errmsg = f"Reading of input to 'prune_by_energy' failed: {e}."
            logging.error(f"'{self.namespace}' | {_called_from}()\n{_errmsg}")
            raise RuntimeError(f"{_called_from}(): {_errmsg}")

        if error_message is not None:
            logging.error(f"'{self.namespace}' | {_called_from}()\n{error_message}")
            raise ValueError(f"{_called_from}(): {error_message}.")

        # Get relative energies
        rel_energies = self._get_relative_energies()

        # Loop over energies and prune ensemble
        for conf_idx, rel_energy in enumerate(rel_energies):
            if rel_energy > cutoff_value_:
                self.is_valid[conf_idx] = False
                logging.info(
                    f"'{self.namespace}' | {_called_from}()\nConformer pruning: the conformer with "
                    f"index {conf_idx} has a relative energy of {round(rel_energy, 2)} kJ/mol and "
                    "was therefore set to be invalid."
                )
            else:
                logging.info(
                    f"'{self.namespace}' | {_called_from}()\nConformer pruning: the conformer with "
                    f"index {conf_idx} has a relative energy of {round(rel_energy, 2)} kJ/mol and "
                    "was therefore kept valid."
                )

    def compare_conformers(self) -> None:
        """Check if all conformers in the molecule vault are identical by substructure matching.

        This is done by comparing all conformers to the first conformer in the molecule vault. If
        a mismatch is found, a warning is logged but no further actions are taken. However, such
        a mismatch is detrimental for many downstream tasks.

        Returns
        -------
        None
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        _found_mismatch = False

        ref_mol = self.mol_objects[0]
        ref_n_atoms = ref_mol.GetNumAtoms()

        assert isinstance(self.size, int)  # for type checker
        for idx in range(1, self.size):
            match = ref_mol.GetSubstructMatch(self.mol_objects[idx])
            if len(match) != ref_n_atoms:
                _found_mismatch = True
                logging.warning(
                    f"'{self.namespace}' | {_loc}()\nThe conformer with index {idx} does not match "
                    "the conformer with index 0. Ensure that all conformers represent the same "
                    "molecule and that this mismatch will not break downstream tasks."
                )

        if _found_mismatch is False:
            logging.info(
                f"'{self.namespace}' | {_loc}()\nAll conformers in the molecule vault are "
                "identical as determined by substructure matching."
            )

    def clean_properties(self) -> None:
        """Remove undesired properties from the atom and bond objects of the molecule objects.

        Returns
        -------
        None
        """
        for mol in self.mol_objects:
            # Atoms
            for atom in mol.GetAtoms():
                for prop in UNDESIRED_ATOM_BOND_PROPERTIES:
                    if atom.HasProp(prop):
                        atom.ClearProp(prop)

            # Bonds
            for bond in mol.GetBonds():
                for prop in UNDESIRED_ATOM_BOND_PROPERTIES:
                    if bond.HasProp(prop):
                        bond.ClearProp(prop)

    def clear_feature_cache_(self, feature_type: str, origins: Optional[List[str]]) -> None:
        """Remove cached feature data from the individual atom and bond feature caches.

        The ``feature_type`` and ``origins``parameters define which cached features are removed. If
        ``origins`` is ``None``, all cached features are removed. For atoms, the caches for the
        actual molecule, the one-electron-oxidized molecule, and the one-electron-reduced molecule
        are cleared.

        Cached global features are always all removed when this method is called.

        Parameters
        ----------
        feature_type : str
            The type of the feature(s) to be cleared, either "atom" or "bond".
        origins : Optional[List[str]]
            A list of the names of the feature origins to be cleared. If ``None``, all cached
            features are removed.

        Returns
        -------
        None
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Format size of molecule vault
        assert isinstance(self.size, int)  # for type checker
        if self.input_type == "smiles":
            _size = 1
        else:
            _size = self.size

        # Atoms
        if feature_type == "atom":
            for idx in range(_size):
                # Remove all data
                if origins is None:
                    self.atom_feature_cache_n[idx] = {}
                    self.atom_feature_cache_n_minus1[idx] = {}
                    self.atom_feature_cache_n_plus1[idx] = {}

                    # Remove properties from the atom objects
                    for atom in self.mol_objects[idx].GetAtoms():
                        for prop in list(atom.GetPropNames()):
                            atom.ClearProp(prop)

                    _infmsg = (
                        f"'{self.namespace}' | {_loc}()\nAll cached 'atom' features were removed"
                    )
                    if self.dimensionality == "2D":
                        _infmsg += "."
                    else:
                        _infmsg += f" for conformer with index {idx}."

                # Remove selected data
                else:
                    for origin in origins:
                        _pattern = rf"{origin}(2|3)D-{feature_type}-"

                        # Cache of the actual molecule
                        self.atom_feature_cache_n[idx] = {
                            name: data
                            for name, data in self.atom_feature_cache_n[idx].items()
                            if not re.match(_pattern, name)
                        }

                        # Cache of the one-electron-oxidized molecule
                        self.atom_feature_cache_n_minus1[idx] = {
                            name: data
                            for name, data in self.atom_feature_cache_n_minus1[idx].items()
                            if not re.match(_pattern, name)
                        }

                        # Cache of the one-electron-reduced molecule
                        self.atom_feature_cache_n_plus1[idx] = {
                            name: data
                            for name, data in self.atom_feature_cache_n_plus1[idx].items()
                            if not re.match(_pattern, name)
                        }

                        # Mol objects
                        for atom in self.mol_objects[idx].GetAtoms():
                            for prop in list(atom.GetPropNames()):
                                if re.match(_pattern, prop):
                                    atom.ClearProp(prop)

                        _infmsg = (
                            f"'{self.namespace}' | {_loc}()\n'Atom' features calculated "
                            f"with '{origin}' were removed"
                        )
                        if self.dimensionality == "2D":
                            _infmsg += "."
                        else:
                            _infmsg += f" for conformer with index {idx}."

        # Bonds
        if feature_type == "bond":
            for idx in range(_size):
                # Remove all data
                if origins is None:
                    self.bond_feature_cache[idx] = {}

                    # Remove properties from the bond objects
                    for bond in self.mol_objects[idx].GetBonds():
                        for prop in list(bond.GetPropNames()):
                            bond.ClearProp(prop)

                    _infmsg = (
                        f"'{self.namespace}' | {_loc}()\nAll cached 'bond' features were removed"
                    )
                    if self.dimensionality == "2D":
                        _infmsg += "."
                    else:
                        _infmsg += f" for conformer with index {idx}."

                # Remove selected data
                else:
                    for origin in origins:
                        _pattern = rf"{origin}(2|3)D-{feature_type}-"

                        # Cache of the actual molecule (other caches don't exist for bonds)
                        self.bond_feature_cache[idx] = {
                            name: data
                            for name, data in self.bond_feature_cache[idx].items()
                            if not re.match(_pattern, name)
                        }

                        # Mol objects
                        for bond in self.mol_objects[idx].GetBonds():
                            for prop in list(bond.GetPropNames()):
                                if re.match(_pattern, prop):
                                    bond.ClearProp(prop)

                        _infmsg = (
                            f"'{self.namespace}' | {_loc}()\n'Bond' features calculated "
                            f"with '{origin}' were removed"
                        )
                        if self.dimensionality == "2D":
                            _infmsg += "."
                        else:
                            _infmsg += f" for conformer with index {idx}."

        logging.info(_infmsg)

        # Global features
        for idx in range(_size):
            self.global_feature_cache[idx] = {}

            _infmsg = f"'{self.namespace}' | {_loc}()\nAll cached 'global' features were removed"
            if self.dimensionality == "2D":
                _infmsg += "."
            else:
                _infmsg += f" for conformer with index {idx}."

        logging.info(_infmsg)

    def update_boltzmann_weights(
        self, temperature: Union[float, int], ignore_invalid: bool
    ) -> None:
        """Update the ``boltzmann_weights`` attribute of the ``MolVault`` object based on
        ``energies_n`` by calculating the Boltzmann weights at a given temperature.

        Parameters
        ----------
        temperature : Union[float, int]
            The temperature in Kelvin at which the Boltzmann weights are computed.
        ignore_invalid : bool
            If ``True``, invalid conformers will be ignored in the calculation, if ``False``,
            weights will not be computed for ensembles with mixed valid/invalid conformers and
            all weights will be set to ``None``.

        Returns
        -------
        None
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Check for invalid conformers
        _invalid_counter = 0
        for conf_idx, valid in enumerate(self.is_valid):
            if valid is False and ignore_invalid is True:
                logging.warning(
                    f"'{self.namespace}' | {_loc}()\nThe conformer with index {conf_idx} is "
                    "invalid and the ignoring of invalid conformers was requested. Therefore, "
                    "this conformer is ignored in the calculation of the Boltzmann weights and "
                    "will have a Boltzmann weight of None."
                )
                _invalid_counter += 1

            elif valid is False and ignore_invalid is False:
                logging.warning(
                    f"'{self.namespace}' | {_loc}()\nThe conformer with index {conf_idx} is "
                    "invalid and the ignoring of invalid conformers was not requested. Therefore, "
                    "Boltzmann weights will not be computed and will be set to None for "
                    "all conformers."
                )
                assert isinstance(self.size, int)  # for type checker
                self.boltzmann_weights = (temperature, [None] * self.size)
                return

        # Check if all conformers are invalid
        if set(self.is_valid) == {False}:
            logging.warning(
                f"'{self.namespace}' | {_loc}()\nAll conformers in the ensemble are invalid. "
                "Therefore, Boltzmann weights will be set to None for all conformers."
            )
            assert isinstance(self.size, int)  # for type checker
            self.boltzmann_weights = (temperature, [None] * self.size)
            return

        rel_energies = self._get_relative_energies()
        RT = R * temperature / 1000  # J -> kJ

        # Compute Boltzmann weights for all valid conformers
        weights = np.exp(-rel_energies / RT)
        weights = np.where(np.array(self.is_valid), weights, np.nan)
        weights /= np.nansum(weights)

        # Finally, convert NaN back to None for the list
        weights_list = [None if np.isnan(w) else float(w) for w in weights]

        # Set weights
        self.boltzmann_weights = (temperature, weights_list)

        _formatted_weights = [
            round(w, 5) if type(w) == float else w for w in self.boltzmann_weights[1]
        ]
        logging.info(
            f"'{self.namespace}' | {_loc}()\nBoltzmann weights were calculated at "
            f"{temperature} K. {_invalid_counter} out of the {self.size} conformers were invalid "
            f"and were ignored in the calculation.\nBoltzmann weights: {_formatted_weights}."
        )

    def _render_mol_3D(
        self, mol_blocks: List[str], idx_type: Optional[str], image_size: Tuple[int, int]
    ) -> ipywidgets.VBox:
        """Render an interactive 3D view of one or an ensemble of conformers in a Jupyter
        notebook with optional atom or bond indices added to the structure.

        Parameters
        ----------
        mol_blocks : List[str]
            A list of MOL blocks for all conformers in the molecule vault.
        idx_type : Optional[str]
            The type of indices to add to the structure, either "atom", "bond", or ``None``.
        image_size : Tuple[int, int]
            The size of the generated image in pixels as a 2-tuple.

        Returns
        -------
        ipywidgets.VBox
            A VBox widget containing the interactive 3D viewer, a slider to select the conformer,
            and printed information about the currently displayed conformer.
        """
        # Get widgets
        assert isinstance(self.size, int)  # for type checker

        viewer_output = ipywidgets.Output()
        print_output = ipywidgets.Output()
        slider = ipywidgets.IntSlider(
            value=0,
            min=0,
            max=self.size - 1,
            step=1,
            description="Conformer index:",
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style={"description_width": "100px"},
            layout=ipywidgets.Layout(width="600px"),
        )

        def _make_view(conformer_idx: int) -> None:
            """Create and display a 3D viewer for a specific conformer."""
            mol = self.mol_objects[conformer_idx]

            # Create 3D viewer
            viewer = py3Dmol.view(width=image_size[0], height=image_size[1])
            viewer.addModel(mol_blocks[conformer_idx], "mol")
            viewer.setStyle({"stick": {"colorscheme": "default"}})

            # Add atom or bond indices as labels if requested
            if idx_type is None:
                pass

            elif idx_type == "atom":
                for atom in mol.GetAtoms():
                    atom_idx = atom.GetIdx()
                    pos = mol.GetConformer().GetAtomPosition(atom_idx)

                    viewer.addLabel(
                        str(atom_idx),
                        {
                            "position": {"x": pos.x, "y": pos.y, "z": pos.z},
                            "showBackground": False,
                            "fontColor": "black",
                            "fontSize": max(image_size) // 27,
                        },
                    )

            elif idx_type == "bond":
                for bond in mol.GetBonds():
                    bond_idx = bond.GetIdx()
                    pos1 = mol.GetConformer().GetAtomPosition(bond.GetBeginAtomIdx())
                    pos2 = mol.GetConformer().GetAtomPosition(bond.GetEndAtomIdx())
                    mid_pos = {
                        "x": (pos1.x + pos2.x) / 2,
                        "y": (pos1.y + pos2.y) / 2,
                        "z": (pos1.z + pos2.z) / 2,
                    }

                    viewer.addLabel(
                        str(bond_idx),
                        {
                            "position": mid_pos,
                            "showBackground": False,
                            "fontColor": "black",
                            "fontSize": max(image_size) // 27,
                        },
                    )

            viewer.zoomTo()
            viewer.show()

        def _make_print(conformer_idx: int) -> None:
            """Print information for a specific conformer."""
            print(f"Is valid:           {self.is_valid[conformer_idx]}")
            if self.energies_n_read is True:
                rel_energies = self._get_relative_energies()
                rel_energy = rel_energies[conformer_idx]
                print(f"Relative energy:    {round(rel_energy, 2)} kJ/mol")
                print(
                    f"Energy rank index:  {np.argsort(rel_energies).tolist().index(conformer_idx)}"
                )

            if len(self.boltzmann_weights) != 0:
                assert self.boltzmann_weights[1] is not None  # for type checker
                weight = self.boltzmann_weights[1][conformer_idx]
                if weight is not None:
                    print(
                        f"Boltzmann weight:   {round(weight, 5)} (at {self.boltzmann_weights[0]} K)"
                    )
                else:
                    print(f"Boltzmann weight:   {weight}")

        def _update_conformer(change_request: traitlets.utils.bunch.Bunch) -> None:
            """Update the viewer and the print output to display new information based on the
            slider value.
            """
            conformer_idx = change_request["new"]

            with viewer_output:
                viewer_output.clear_output(wait=True)
                _make_view(conformer_idx=conformer_idx)

            with print_output:
                print_output.clear_output(wait=True)
                _make_print(conformer_idx=conformer_idx)

        def _init_viewer() -> None:
            """Initialize the viewer and the print output."""
            time.sleep(0.5)  # Required for visualizing vaults with only one conformer

            with viewer_output:
                viewer_output.clear_output(wait=True)
                _make_view(conformer_idx=0)

            with print_output:
                print_output.clear_output(wait=True)
                _make_print(conformer_idx=0)

        # Build controls
        slider.observe(_update_conformer, names="value")
        controls = ipywidgets.VBox([slider, print_output, viewer_output])

        # Initialize the viewer and print output (done in a separate thread to work-around a js
        # error that can pop up in VSCode notebooks)
        thread = threading.Thread(target=_init_viewer, daemon=True)
        thread.start()
        thread.join(timeout=5.0)

        return controls

    def _get_relative_energies(self) -> NDArray[np.float64]:
        """Get the relative energies of the conformers in kJ/mol.

        Returns
        -------
        NDArray[np.float64]
            The relative energies in kJ/mol.
        """
        energies = np.array([energy for energy, _ in self.energies_n], dtype=float)
        min_energy = np.nanmin(energies)
        rel_energies: NDArray[np.float64] = energies - min_energy
        return rel_energies

    @staticmethod
    def _extract_energy_from_xyz_block(
        xyz_block: str,
    ) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]:
        """Read the energy from the second line of an XYZ block.

        If the energy cannot be extracted, ``None`` is returned.

        Parameters
        ----------
        xyz_block : str
            The XYZ block as a string.

        Returns
        -------
        Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]
            A tuple containing

            * the energy as submitted,
            * the unit as submitted,
            * the new energy in kJ/mol, and
            * an error message.

            The error message is ``None`` if the extraction was successful.
        """
        second_line = xyz_block.split("\n")[1] + "\n"
        energy_as_submitted, unit_as_submitted, energy, error_message = extract_energy_from_string(
            line=second_line
        )
        return energy_as_submitted, unit_as_submitted, energy, error_message

    @staticmethod
    def _extract_energy_from_mol_object(
        mol: Chem.rdchem.Mol,
    ) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]:
        """Read the energy from the properties of an RDKit molecule object.

        The energy is expected to be stored under the property name "energy".

        Parameters
        ----------
        mol : Chem.rdchem.Mol
            The RDKit molecule object.

        Returns
        -------
        Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]
            A tuple containing

            * the energy as submitted,
            * the unit as submitted,
            * the new energy in kJ/mol, and
            * an error message.

            The error message is ``None`` if the extraction was successful.
        """
        energy_as_submitted = None
        unit_as_submitted = None
        energy = None
        _errmsg = None
        _found = False

        for prop, value in mol.GetPropsAsDict().items():
            prop = standardize_string(inp_data=prop)
            if prop == "energy":
                _found = True
                energy_as_submitted, unit_as_submitted, energy, _errmsg = (
                    extract_energy_from_string(line=value)
                )
                break

        if _found is False:
            _errmsg = "no property named 'energy' was found in the RDKit mol object"

        return energy_as_submitted, unit_as_submitted, energy, _errmsg
