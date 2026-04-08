"""BONAFIDE main module."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem

from bonafide._bonafide import _AtomBondFeaturizer
from bonafide.utils.constants import (
    DETERMINE_BONDS_METHODS,
    ELECTRONIC_STRUCTURE_ENGINES,
    FEATURE_TYPES,
    INPUT_FILE_EXTENSIONS,
    INPUT_TYPES,
    REDOX_STATES,
    REDOX_STATES2,
)
from bonafide.utils.custom_featurizer_input_validation import custom_featurizer_data_validator
from bonafide.utils.feature_factories import FEATURE_FACTORIES
from bonafide.utils.helper_functions import get_function_or_method_name, standardize_string
from bonafide.utils.io_ import read_mol_object, read_sd_file, read_xyz_file
from bonafide.utils.molecule_vault import MolVault
from bonafide.utils.string_formatting import (
    _make_bold_end,
    _make_bold_start,
    _make_green_end,
    _make_green_start,
    _make_underlined_end,
    _make_underlined_start,
)

if TYPE_CHECKING:
    import ipywidgets
    from PIL import PngImagePlugin


class AtomBondFeaturizer(_AtomBondFeaturizer):
    """Main class of the Bond and Atom Featurizer and Descriptor Extractor (BONAFIDE).

    It implements all the methods available to the user to calculate atom and or bond-specific
    features.

    Parameters
    ----------
    log_file_name : str, optional
        The name of the log file to which all logging messages are written, by default
        "bonafide.log". A file with this name cannot already exists.

    Attributes
    ----------
    _atom_feature_indices_2D : List[int]
        The list of atom feature indices that can be calculated for molecules for which only 2D
        information is available.
    _atom_feature_indices_3D : List[int]
        The list of atom feature indices that can be calculated for molecules for which 3D
        information is available.
    _bond_feature_indices_2D : List[int]
        The list of bond feature indices that can be calculated for molecules for which only 2D
        information is available.
    _bond_feature_indices_3D : List[int]
        The list of bond feature indices that can be calculated for molecules for which 3D
        information is available.
    _feature_config : Dict[str, Any]
        The configuration settings for the individual programs used for feature calculation. The
        default settings are loaded from the ``_feature_config.toml`` file. The current settings
        can be inspected with the ``print_options()`` method and changed using the
        ``set_options()`` method.
    _feature_info : Dict[int, Dict[str, Any]]
        The metadata of all implemented atom and bond features, e.g., the name of the feature, its
        dimensionality requirements (either 2D or 3D), or the program it is calculated with
        (origin). The data is loaded from the ``_feature_info.json`` file and should not be
        manually modified.
    _feature_info_df : pd.DataFrame
        A pandas DataFrame containing the feature indices (as index of the DataFrame) and their key
        characteristics of all implemented atom and bond features.
    _functional_groups_smarts : Dict[str, List[Tuple[str, Chem.rdchem.Mol]]]
        A dictionary containing the names and SMARTS patterns of different functional groups.
    _init_directory : str
        The path to the directory where the ``AtomBondFeaturizer`` object was initialized.
    _keep_output_files : bool
        If ``True``, all output files created during the feature calculations are kept. If
        ``False``, they are removed when the calculation is done.
    _loc : str
        The location string representing the current class and method for logging purposes.
    _namespace : Optional[str]
        The namespace for the molecule as defined by the user when reading in the molecule.
    _output_directory : Optional[str]
        The path to the directory where all output files created during the feature calculations
        are stored (if requested).
    _periodic_table : Dict[str, element]
        A dictionary representing the periodic table with element symbols as keys and mendeleev
        ``element`` objects as values.
    mol_vault : Optional[MolVault]
        Dataclass object for storing all relevant data on the molecule for which features should be
        calculated.
    """

    def __init__(self, log_file_name: str = "bonafide.log") -> None:
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Initialize attributes
        self._feature_config = {}
        self._feature_info = {}
        self._feature_info_df = pd.DataFrame()
        self._atom_feature_indices_2D = []
        self._bond_feature_indices_2D = []
        self._atom_feature_indices_3D = []
        self._bond_feature_indices_3D = []
        self._periodic_table = {}
        self._functional_groups_smarts = {}

        self.mol_vault = None  # type: ignore[assignment]
        self._namespace = None
        self._keep_output_files = False
        self._output_directory = None
        self._init_directory = os.getcwd()

        # Initialize logging
        self._init_logging(log_file_name=log_file_name)

        # Load configuration toml file
        self._load_config_file()

        # Load feature file
        self._load_feature_info_file()

    def list_atom_features(self, **kwargs: Any) -> pd.DataFrame:
        """Display all available atom features.

        The DataFrame can be filtered with the following optional keyword arguments:

        * name
        * origin
        * dimensionality
        * data_type
        * requires_electronic_structure_data
        * requires_bond_data
        * requires_charge
        * requires_multiplicity
        * config_path
        * factory

        Parameters
        ----------
        **kwargs : Any
            Additional optional keyword arguments for filtering the feature DataFrame. If empty,
            all atom features are returned.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the selected atom features and their characteristics.
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        if "feature_type" in kwargs:
            _errmsg = "Invalid input: 'feature_type' is set automatically and must not be provided."
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        return self._list_features(feature_type="atom", **kwargs)

    def list_bond_features(self, **kwargs: Any) -> pd.DataFrame:
        """Display all available bond features.

        The DataFrame can be filtered with the following optional keyword arguments:

        * name
        * origin
        * dimensionality
        * data_type
        * requires_electronic_structure_data
        * requires_bond_data
        * requires_charge
        * requires_multiplicity
        * config_path
        * factory

        Parameters
        ----------
        **kwargs : Any
            Additional optional keyword arguments for filtering the feature DataFrame. If empty,
            all bond features are returned.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the selected bond features and their characteristics.
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        if "feature_type" in kwargs:
            _errmsg = "Invalid input: 'feature_type' is set automatically and must not be provided."
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        return self._list_features(feature_type="bond", **kwargs)

    def print_options(self, origin: Optional[Union[str, List[str]]] = None) -> None:
        """Print the configuration settings of the individual programs for feature calculation.

        By providing input to the ``origin`` parameter, it can be selected which program's settings
        are printed. Valid origins are:

        * alfabet
        * bonafide
        * dbstep
        * dscribe
        * kallisto
        * mendeleev
        * morfeus
        * multiwfn
        * psi4
        * qmdesc
        * rdkit
        * xtb

        Parameters
        ----------
        origin : Optional[Union[str, List[str]]], optional
            The name(s) of the program(s) for which the configuration settings should be printed.
            Can either be given as string or list of multiple programs, by default ``None``. If
            kept ``None``, the settings of all programs are printed.

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(f"'{self._namespace}' | {self._loc}() | START\n> 'origin':  {origin}\n-----")

        # Format origin input
        if len(self._feature_config) == 0:
            _errmsg = (
                "The configuration settings cannot be printed because the default settings could "
                "not be loaded."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        _valid_origins = list(self._feature_config.keys())
        _inpt = type(origin)
        if origin is None:
            _for_printing = self._feature_config

        elif isinstance(origin, str):
            self._check_is_str_in_list(
                parameter_name="origin", value=origin, allowed_values=_valid_origins
            )
            _for_printing = {origin: self._feature_config[origin]}

        elif isinstance(origin, list):
            origin = [standardize_string(inp_data=o) for o in origin]
            for o in origin:
                self._check_is_str_in_list(
                    parameter_name="origin", value=o, allowed_values=_valid_origins
                )
            _for_printing = {o: self._feature_config[o] for o in origin}

        else:
            _errmsg = (
                "Invalid input to 'origin': must be either None, of type str, or a list of "
                f"strings but obtained {_inpt.__name__}."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise TypeError(f"{self._loc}(): {_errmsg}")

        # Print data
        _file_name = "_feature_config.toml"
        _toml_config_file_path = os.path.join(os.path.dirname(__file__), _file_name)
        print(
            f"{_make_underlined_start}{_make_bold_start}Default configuration settings at:"
            f"{_make_bold_end}{_make_underlined_end}"
        )
        print(_toml_config_file_path)
        print()

        for orig, data in _for_printing.items():
            print(
                f"{_make_underlined_start}{_make_bold_start}{_make_green_start}{orig}"
                f"{_make_green_end}{_make_bold_end}{_make_underlined_end}"
            )
            for d, vals in data.items():
                print(f"    {_make_bold_start}{d}{_make_bold_end}: {vals}")
            print()

        logging.info(f"'{self._namespace}' | {self._loc}()\nConfiguration settings were printed.")
        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

    def set_options(self, configs: Union[Tuple[str, Any], List[Tuple[str, Any]]]) -> None:
        """Change configuration settings for the individual programs used for feature calculation.

        The input to this method must be a 2-tuples (or a list thereof), where the first entry is
        the path to the configuration setting that should be changed (point-separated) and the
        second entry is the new value.

        For listing all available configuration settings and their current values, see the
        ``print_options()`` method.

        Parameters
        ----------
        configs : Union[Tuple[str, Any], List[Tuple[str, Any]]]
            A 2-tuple or a list of 2-tuples containing the configuration paths and their new
            values, e.g.: ("bonafide.autocorrelation.depth", 3)

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(f"'{self._namespace}' | {self._loc}() | START\n> 'configs':  {configs}\n-----")

        # Check input types
        self._check_is_of_type(expected_type=[tuple, list], value=configs, parameter_name="configs")
        if isinstance(configs, tuple):
            configs = [configs]

        # Check if list is empty
        if len(configs) == 0:
            _errmsg = "Invalid input to 'configs': must not be an empty list."
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check individual entries in list
        for entry in configs:
            self._check_is_of_type(
                expected_type=tuple,
                value=entry,
                parameter_name="configs",
                prefix="each element of the list",
            )

            if len(entry) != 2:
                _errmsg = (
                    f"Invalid input to 'configs': must be a 2-tuple or a list of 2-tuples but "
                    f"obtained a list containing a tuple of length {len(entry)}."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

        # Iterate over all configuration changes
        for config_path, value in configs:
            self._check_is_of_type(
                expected_type=str,
                value=config_path,
                parameter_name="configs",
                prefix="first entry of each 2-tuple",
            )

            # Make branch of config path case insensitive, actual option name stays case-sensitive
            if len(config_path.split(".")) >= 2:
                _option_name, _branch = config_path[::-1].split(".", 1)
                _branch = standardize_string(inp_data=_branch)
                config_path = f"{_branch[::-1]}.{_option_name[::-1]}"

            # Apply changes
            self._set_options(config_path, value)

        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

    def read_input(
        self,
        input_value: Union[str, Chem.rdchem.Mol],
        namespace: str,
        input_format: str = "smiles",
        read_energy: bool = False,
        prune_by_energy: Optional[Tuple[Union[int, float], str]] = None,
        output_directory: Optional[str] = None,
    ) -> None:
        """Read in a SMILES string, an input file (either XYZ or SDF), or an RDKit molecule object.

        By default, the ``input_format`` parameter is set to "smiles", meaning that a SMILES string
        can be passed to the method without specifying ``input_format``. If a file should be read
        in, ``input_format`` must be set to "file"; for an RDKit molecule object, it must be set to
        "mol_object".

        If it is intended to read in energies from the input file or the RDKit molecule object (if
        available), the ``read_energy`` parameter must be set to ``True``. This will set the
        energies in the molecule vault for state "n" (actual molecule). Alternatively, the
        ``attach_energy()`` method can be used to attach energy data to the molecule vault after
        reading in the molecule. This method also allows to attach energies for different redox
        states ("n" (actual molecule), "n+1" (one-electron reduced molecule), "n-1"
        (one-electron oxidized molecule)).

        Energy data must always be specified as strings containing the value and the respective
        unit separated by a space, for example, ``"-10.5 kcal/mol"`` or ``"-1254.21548 Eh"``.
        Supported energy units are "Eh", "kcal/mol", and "kJ/mol".

        It is possible to prune the conformer ensemble through the ``prune_by_energy`` parameter.
        Pruning is done based on relative energies (of state "n") with respect to the lowest-energy
        conformer in the molecule vault.

        Passing an input to ``output_directory`` allows to specify where all output files created
        during the feature calculations are stored. If kept ``None``, all output files are deleted.

        Parameters
        ----------
        input_value : Union[str, Chem.rdchem.Mol]
            The path to the input file, a SMILES string, or an RDKit molecule object.
        namespace : str
            The namespace for the molecule that is read in. This identifier is used throughout all
            following BONAFIDE processes including logging.
        input_format : str, optional
            The type of input. Can either be "file" or "smiles", by default "smiles".
        read_energy : bool, optional
            If ``True``, it is attempted to read in energies from the input file (if available), by
            default ``False``. These energies are set for state "n" (actual molecule).
        prune_by_energy : Optional[Tuple[Union[int, float], str]], optional
            If a value other than ``None`` is provided, all conformers with a relative energy above
            this value are set to be invalid and ignored during feature calculation and any
            further processing. The input must be a 2-tuple in which the first entry is the
            relative energy cutoff value and the second entry is the respective energy unit.
            Supported units are "Eh", "kcal/mol", and "kJ/mol". If ``None``, no pruning is
            performed, by default ``None``.
        output_directory : Optional[str], optional
            The path to the directory where all output files created during the feature
            calculations are stored. If kept ``None``, no output files folder is created and all
            output files are deleted after data extraction.

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(
            f"'{namespace}' | {self._loc}() | START\n"
            f"> 'input_value':       {input_value}\n"
            f"> 'namespace':         {namespace}\n"
            f"> 'input_format':      {input_format}\n"
            f"> 'read_energy':       {read_energy}\n"
            f"> 'prune_by_energy':   {prune_by_energy}\n"
            f"> 'output_directory':  {output_directory}\n"
            "-----"
        )

        # Check input types
        # The input to prune_by_energy is checked in MolVault.prune_ensemble_by_energy()
        self._check_is_of_type(expected_type=str, value=namespace, parameter_name="namespace")
        namespace = namespace.strip()
        self._namespace = namespace

        self._check_is_of_type(
            expected_type=[str, Chem.rdchem.Mol], value=input_value, parameter_name="input_value"
        )
        self._check_is_of_type(expected_type=str, value=input_format, parameter_name="input_format")
        self._check_is_of_type(expected_type=bool, value=read_energy, parameter_name="read_energy")

        # Check if string input to input_value is empty
        if isinstance(input_value, str) and input_value.strip() == "":
            _errmsg = "Invalid input to 'input_value': must not be an empty string."
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check input to input_format
        input_format = self._check_is_str_in_list(
            parameter_name="input_format", value=input_format, allowed_values=INPUT_TYPES
        )

        # Reset output directory related attributes in case another molecule was already
        # loaded before
        self._keep_output_files = False
        self._output_directory = None

        # Handle output_directory input
        if output_directory is None:
            logging.warning(
                f"'{self._namespace}' | {self._loc}()\nThe input to 'output_directory' is None "
                "(default value). Therefore, all output files potentially generated during the "
                "feature calculations will not be kept. Provide a valid input to "
                "'output_directory' if the output files should be stored."
            )
            self._output_directory = os.getcwd()
        else:
            self._check_is_of_type(
                expected_type=str, value=output_directory, parameter_name="output_directory"
            )
            self._setup_output_directory(dir_path=output_directory)

        # Handle file input
        if input_format == "file":
            file_type = os.path.splitext(input_value)[-1][1:]

            # Check file extension
            file_type = self._check_is_str_in_list(
                parameter_name="input_value", value=file_type, allowed_values=INPUT_FILE_EXTENSIONS
            )

            # Check if path points to a file
            if not os.path.isfile(input_value):
                _errmsg = "Invalid input to 'input_value': path to the input file is invalid."
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise FileNotFoundError(f"{self._loc}(): {_errmsg}")

            mol_inputs: Any

            # Read XYZ file
            if file_type == "xyz":
                mol_inputs, error_message = read_xyz_file(file_path=input_value)
                if error_message is not None:
                    _errmsg = f"Reading data from XYZ file failed: {error_message}."
                    logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                    raise ValueError(f"{self._loc}(): {_errmsg}")
                logging.info(
                    f"'{self._namespace}' | {self._loc}()\nReading data from XYZ file successful."
                )

            # Read SD file
            if file_type == "sdf":
                mol_inputs, error_message, stereo_message = read_sd_file(file_path=input_value)

                if error_message is not None:
                    _errmsg = f"Reading data from SD file failed: {error_message}."
                    logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                    raise ValueError(f"{self._loc}(): {_errmsg}")

                if stereo_message is not None:
                    logging.warning(f"'{self._namespace}' | {self._loc}()\n{stereo_message}")

                logging.info(
                    f"'{self._namespace}' | {self._loc}()\nReading data from SD file successful."
                )

            input_type = file_type

        # Handle RDKit molecule object input
        elif input_format == "mol_object":
            input_type = input_format
            init_mol, processed_mols, error_message = read_mol_object(input_value)
            if error_message is not None:
                _errmsg = f"Reading data from RDKit mol object failed: {error_message}."
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise TypeError(f"{self._loc}(): {_errmsg}")
            mol_inputs = (init_mol, processed_mols)
            logging.info(
                f"'{self._namespace}' | {self._loc}()\nReading data from RDKit molecule object "
                "successful."
            )

        # Handle SMILES input
        else:
            input_type = "smiles"
            mol_inputs = [input_value]
            logging.info(
                f"'{self._namespace}' | {self._loc}()\nReading data from SMILES string successful."
            )

        # Reset molecule vault attribute in case another molecule was already loaded before
        # (just to make sure)
        self.mol_vault = None  # type: ignore[assignment]

        # Initialize molecule vault
        self.mol_vault = MolVault(
            mol_inputs=mol_inputs,
            namespace=namespace,
            input_type=input_type,
        )
        self.mol_vault.initialize_mol()
        self.mol_vault.get_elements()

        # Read energy from file if requested
        if read_energy is True and input_format in ["file", "mol_object"]:
            self.mol_vault.read_mol_energies()

        # Prune conformer ensemble if requested
        if prune_by_energy is not None and input_format in ["file", "mol_object"]:
            self.mol_vault.prune_ensemble_by_energy(
                energy_cutoff=prune_by_energy, _called_from=self._loc
            )

        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

    def show_molecule(
        self,
        index_type: Optional[str] = "atom",
        in_3D: bool = False,
        image_size: Tuple[int, int] = (500, 500),
    ) -> Union[PngImagePlugin.PngImageFile, ipywidgets.VBox]:
        """Display the molecule with atom, bond or no indices.

        Molecules can either be shown in an interactive 3D view (if 3D information is available)
        or in 2D as a Lewis structure.

        Parameters
        ----------
        index_type : str, optional
            The type of indices to add to the structure, either "atom", "bond", or ``None``. By
            default "atom".
        in_3D : bool, optional
            If ``True``, the molecule is shown in 3D (if 3D information is available), by default
            ``False``.
        image_size : Tuple[int, int], optional
            The size of the displayed image in pixels (width, height), by default (500, 500).

        Returns
        -------
        Union[PngImagePlugin.PngImageFile, ipywidgets.VBox]
            A 2D or 3D depiction of the molecule, either as an image or an interactive 3D view.
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(
            f"'{self._namespace}' | {self._loc}() | START\n"
            f"> 'index_type':  {index_type}\n"
            f"> 'in_3D':       {in_3D}\n"
            f"> 'image_size':  {image_size}\n"
            "-----"
        )

        # Pre-checks
        self._check_is_initialized(error_message="showing the molecule")

        # Check input types
        self._check_is_of_type(
            expected_type=[str, None], value=index_type, parameter_name="index_type"
        )
        self._check_is_of_type(expected_type=bool, value=in_3D, parameter_name="in_3D")
        self._check_is_of_type(expected_type=tuple, value=image_size, parameter_name="image_size")

        # Check input to index_type
        _helper_list = [x for x in FEATURE_TYPES]
        _helper_list.append("none")
        index_type = self._check_is_str_in_list(
            parameter_name="index_type", value=index_type, allowed_values=_helper_list
        )

        # Don't allow 3D rendering if no 3D information is available
        if in_3D is True and self.mol_vault.dimensionality == "2D":
            _errmsg = "A molecule cannot be rendered in 3D if no 3D information is available."
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check size input
        if len(image_size) != 2:
            _errmsg = (
                f"Invalid input to 'image_size': must be a 2-tuple but obtained a tuple of "
                f"length {len(image_size)}."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        for value in image_size:
            if type(value) != int or value <= 0:
                _errmsg = (
                    f"Invalid input to 'image_size': both entries of the 2-tuple must be "
                    f"positive integers but obtained {image_size}."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

        # Show molecule
        disp = self.mol_vault.render_mol(idx_type=index_type, in_3D=in_3D, image_size=image_size)

        if in_3D is True:
            _dimensionality = "3D"
        else:
            _dimensionality = "2D"

        if index_type is None or index_type.lower() == "none":
            _suffix = "."
        else:
            _suffix = f" with '{index_type}' indices."

        logging.info(
            f"'{self._namespace}' | {self._loc}()\nMolecule was rendered in "
            f"{_dimensionality}{_suffix}"
        )
        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

        return disp

    def set_charge(self, charge: int) -> None:
        """Set the charge of the molecule.

        Parameters
        ----------
        charge : int
            The total charge of the molecule that is used for feature calculation.

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(f"'{self._namespace}' | {self._loc}() | START\n> 'charge':  {charge}\n-----")

        # Pre-checks
        self._check_is_initialized(error_message="setting the charge")

        # Check input type
        self._check_is_of_type(expected_type=int, value=charge, parameter_name="charge")

        # Set charge
        self.mol_vault.charge = charge
        logging.info(f"'{self._namespace}' | {self._loc}()\nMolecular charge was set to {charge}.")
        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

    def set_multiplicity(self, multiplicity: int) -> None:
        """Set the multiplicity of the molecule.

        Parameters
        ----------
        multiplicity : int
            The spin multiplicity of the molecule that is used for feature calculation.

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(
            f"'{self._namespace}' | {self._loc}() | START\n> 'multiplicity':  {multiplicity}\n-----"
        )

        # Pre-checks
        self._check_is_initialized(error_message="setting the multiplicity")

        # Check input type
        self._check_is_of_type(expected_type=int, value=multiplicity, parameter_name="multiplicity")

        # Check input
        if multiplicity < 1:
            _errmsg = f"Invalid input to 'multiplicity': must be >= 1 but obtained {multiplicity}."
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Set multiplicity
        self.mol_vault.multiplicity = multiplicity
        logging.info(
            f"'{self._namespace}' | {self._loc}()\nSpin multiplicity was set to {multiplicity}."
        )
        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

    def attach_smiles(
        self,
        smiles: str,
        align: bool = True,
        connectivity_method: str = "connect_the_dots",
        covalent_radius_factor: Union[int, float] = 1.3,
    ) -> None:
        """Attach a SMILES string to a molecule vault that is hosting a 3D molecule.

        Before attaching a SMILES string, the compatibility of the SMILES string with the already
        existing molecule in the vault is checked. The ``align`` parameter allows to decide whether
        to keep the initial atom order (``align=True``) or apply the one of the SMILES string
        (``align=False``).

        The additional optional parameters ``connectivity_method`` and ``covalent_radius_factor``
        influence how the atom connectivity of the RDKit molecule object(s) initially hosted in the
        molecule vault is determined (required for attaching the SMILES string).

        A SMILES string can only be attached to a molecule vault for which the bonds are not
        determined yet. This also means that once a SMILES string is attached to a molecule
        vault, it cannot be changed anymore. A SMILES string cannot be attached to a molecule vault
        hosting a 2D molecule.

        Parameters
        ----------
        smiles : str
            The SMILES string that should be attached to the molecule vault.
        align : bool, optional
            If ``True``, the atom indices of the initially provided 3D structures are preserved, if
            ``False``, the atoms are re-ordered according to the order in the SMILES string, by
            default ``True``.
        connectivity_method : str
            The name of the method that is used to determine the atom connectivity. Available
            options are "connect_the_dots", "van_der_waals", and "hueckel".
        covalent_radius_factor : float
            A scaling factor that is applied to the covalent radii of the atoms when determining the
            bonds with the van-der-Waals method.

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(
            f"'{self._namespace}' | {self._loc}() | START\n"
            f"> 'smiles':                  {smiles}\n"
            f"> 'align':                   {align}\n"
            f"> 'connectivity_method':     {connectivity_method}\n"
            f"> 'covalent_radius_factor':  {covalent_radius_factor}\n"
            "-----"
        )

        # Pre-checks
        self._check_is_initialized(error_message="attaching a SMILES string")

        self._check_is_2D("Attaching a SMILES string to a 2D ensemble is not allowed.")

        # Check if bonds were already determined
        if self.mol_vault.bonds_determined is True:
            _errmsg = (
                "A SMILES string can only be attached to a molecule vault that has its "
                "bonds not yet determined."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check input types
        self._check_is_of_type(expected_type=str, value=smiles, parameter_name="smiles")
        self._check_is_of_type(expected_type=bool, value=align, parameter_name="align")
        self._check_is_of_type(
            expected_type=str, value=connectivity_method, parameter_name="connectivity_method"
        )
        self._check_is_of_type(
            expected_type=[int, float],
            value=covalent_radius_factor,
            parameter_name="covalent_radius_factor",
        )
        covalent_radius_factor = float(covalent_radius_factor)
        if covalent_radius_factor <= 0:
            _errmsg = (
                f"Invalid input to 'covalent_radius_factor': must be > 0 but "
                f"obtained {covalent_radius_factor}."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check connectivity_method input
        connectivity_method = self._check_is_str_in_list(
            parameter_name="connectivity_method",
            value=connectivity_method,
            allowed_values=DETERMINE_BONDS_METHODS,
        )

        # Check if charge is set if Hueckel method is selected
        if connectivity_method == "hueckel" and self.mol_vault.charge is None:
            _errmsg = (
                "Set the charge of the molecule vault with the set_charge() method "
                "before determining bonds with the Hueckel method."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check if features have already been calculated
        _atom_features_calculated = set([len(c) for c in self.mol_vault.atom_feature_cache_n]) != {
            0
        }
        _bond_features_calculated = set([len(c) for c in self.mol_vault.bond_feature_cache]) != {0}

        if any([_atom_features_calculated, _bond_features_calculated]) and align is False:
            _errmsg = (
                'Attaching a SMILES string with "align=False" is not allowed after features '
                "were calculated. This is due to the fact that some features contain atom or bond "
                "indices which would be rendered meaningless when re-ordering of the atoms of the "
                "molecule in the vault is performed."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Attach SMILES
        self._attach_smiles(
            smiles=smiles,
            align=align,
            connectivity_method=connectivity_method,
            covalent_radius_factor=covalent_radius_factor,
        )

        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

    def attach_energy(
        self,
        energy_data: Union[Tuple[Union[int, float], str], List[Tuple[Union[int, float], str]]],
        state: str = "n",
        prune_by_energy: Optional[Tuple[Union[int, float], str]] = None,
    ) -> None:
        """Attach molecular energy values to a molecule vault hosting a 3D molecule.

        The input to ``energy_data`` can either be a single 2-tuple or a list of 2-tuples. Each
        2-tuple must contain the energy value (first entry) and the respective energy unit (second
        entry). Supported energy units are "Eh", "kcal/mol", and "kJ/mol".

        The ``state`` parameter allows to specify to which redox state of the molecule the energy
        values should be attached to.

        If desired, the conformer ensemble can be pruned based on the attached energy values for
        state "n" (actual molecule) through the ``prune_by_energy`` parameter.

        Parameters
        ----------
        energy_data : Union[Tuple[Union[int, float], str], List[Tuple[Union[int, float], str]]]
            A 2-tuple or a list of 2-tuples containing the energy values and respective units.
        state : str, optional
            The redox state of the electronic structure data to be attached, by default "n". Can
            either be

            * "n" (actual molecule),
            * "n+1" (actual molecule plus one electron), or
            * "n-1" (actual molecule minus one electron).
        prune_by_energy : Optional[Tuple[Union[int, float], str]], optional
            If a value other than ``None`` is provided, all conformers with a relative energy above
            this value are set to be invalid and ignored during feature calculation and any
            further processing. The input must be a 2-tuple in which the first entry is the
            relative energy cutoff value and the second entry is the respective energy unit.
            Supported units are "Eh", "kcal/mol", and "kJ/mol". If ``None``, no pruning is
            performed, by default ``None``.
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(
            f"'{self._namespace}' | {self._loc}() | START\n"
            f"> 'energy_data':  {energy_data}\n"
            f"> 'state':        {state}\n"
            "-----"
        )

        # Pre-checks
        self._check_is_initialized(error_message="attaching energy data")

        self._check_is_2D(
            "Attaching molecular energy values to a molecule vault hosting a 2D molecule is "
            "not allowed."
        )

        # Check input types
        # The input to prune_by_energy is checked in MolVault.prune_ensemble_by_energy()
        self._check_is_of_type(
            expected_type=[tuple, list], value=energy_data, parameter_name="energy_data"
        )
        if isinstance(energy_data, tuple):
            energy_data_formatted = [energy_data]
        else:
            energy_data_formatted = [e for e in energy_data]

        for energy_unit_pair in energy_data_formatted:
            self._check_is_of_type(
                expected_type=tuple,
                value=energy_unit_pair,
                parameter_name="energy_data",
                prefix="each entry",
            )

            if len(energy_unit_pair) != 2:
                _errmsg = (
                    f"Invalid entry in 'energy_data': each entry must be a 2-tuple but obtained "
                    f"a tuple of length {len(energy_unit_pair)}."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

        self._check_is_of_type(expected_type=str, value=state, parameter_name="state")

        # Check state input
        state = self._check_is_str_in_list(
            parameter_name="state", value=state, allowed_values=REDOX_STATES2
        )

        # Check that energies matches number of conformers
        if len(energy_data_formatted) != self.mol_vault.size:
            _errmsg = (
                f"The number of provided energy values ({len(energy_data_formatted)}) for state "
                f"'{state}' does not match the number of conformers in the molecule vault "
                f"({self.mol_vault.size})."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Attach energies
        self._attach_energy(energy_data=energy_data_formatted, state=state)

        # Prune conformer ensemble if requested (only based on state "n" energies)
        if prune_by_energy is not None and state == "n":
            self.mol_vault.prune_ensemble_by_energy(
                energy_cutoff=prune_by_energy, _called_from=self._loc
            )
        elif prune_by_energy is not None and state != "n":
            logging.warning(
                f"'{self._namespace}' | {self._loc}()\nPruning the conformer ensemble based on "
                f"energy values for state '{state}' is not supported. No pruning was performed."
            )

        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

    def attach_electronic_structure(
        self, electronic_structure_data: Union[str, List[str]], state: str = "n"
    ) -> None:
        """Attach electronic structure data files to a molecule vault hosting a 3D molecule.

        The input can either be a single file path or a list of file paths. The ``state`` parameter
        allows to specify to which redox state of the molecule the electronic structure data
        should be attached to.

        Parameters
        ----------
        electronic_structure_data : Union[str, List[str]]
            A list of file paths to the electronic structure files or a single file path.
        state : str, optional
            The redox state of the electronic structure data to be attached, by default "n". Can
            either be

            * "n" (actual molecule),
            * "n+1" (actual molecule plus one electron), or
            * "n-1" (actual molecule minus one electron).

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(
            f"'{self._namespace}' | {self._loc}() | START\n"
            f"> 'electronic_structure_data':  {electronic_structure_data}\n"
            f"> 'state':                      {state}\n"
            "-----"
        )

        # Pre-checks
        self._check_is_initialized(error_message="attaching electronic structure data")

        self._check_is_2D(
            "Attaching electronic structure files to a molecule vault hosting a "
            "2D molecule is not allowed."
        )

        # Check input types
        self._check_is_of_type(
            expected_type=[str, list],
            value=electronic_structure_data,
            parameter_name="electronic_structure_data",
        )
        if type(electronic_structure_data) == str:
            el_struc_list = [electronic_structure_data]
        else:
            el_struc_list = [e for e in electronic_structure_data]

        self._check_is_of_type(expected_type=str, value=state, parameter_name="state")

        # Check state input
        state = self._check_is_str_in_list(
            parameter_name="state", value=state, allowed_values=REDOX_STATES2
        )

        # Determine the electronic structure list and types based on the state
        if state == "n+1":
            _el_struc_list = self.mol_vault.electronic_strucs_n_plus1
            _el_struc_types = self.mol_vault.electronic_struc_types_n_plus1
        elif state == "n-1":
            _el_struc_list = self.mol_vault.electronic_strucs_n_minus1
            _el_struc_types = self.mol_vault.electronic_struc_types_n_minus1
        else:
            _el_struc_list = self.mol_vault.electronic_strucs_n
            _el_struc_types = self.mol_vault.electronic_struc_types_n

        # Check if number of electronic structure files matches number of conformers
        if len(el_struc_list) != self.mol_vault.size:
            _errmsg = (
                f"The number of provided electronic structure files ({len(el_struc_list)}) "
                "does not match the number of conformers in the molecule vault "
                f"({self.mol_vault.size})."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        if len(_el_struc_list) != 0:
            _errmsg = (
                f"Electronic structure data for state '{state}' is already attached to the "
                "molecule vault and cannot be attached again."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        self._attach_electronic_structure(
            electronic_struc_list=el_struc_list,
            _el_struc_list=_el_struc_list,  # type: ignore[arg-type]
            _el_struc_types=_el_struc_types,  # type: ignore[arg-type]
            state=state,
        )
        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

    def determine_bonds(
        self,
        connectivity_method: str = "connect_the_dots",
        covalent_radius_factor: Union[int, float] = 1.3,
        allow_charged_fragments: bool = True,
        embed_chiral: bool = True,
    ) -> None:
        """Determine the chemical bonds of each conformer of a molecule vault hosting a
        3D molecule.

        This method can be used to define the chemical bonds of a molecule that was provided
        without information on the bonds (connectivity and bond type). Bond information is
        required for the calculation of certain atom and all bond features.

        The optional parameters ``connectivity_method``, ``covalent_radius_factor``,
        ``allow_charged_fragments``, and ``embed_chiral`` influence how the bonds of the
        individual RDKit molecule object(s) are.

        Parameters
        ----------
        connectivity_method : str
            The name of the method that is used to determine the atom connectivity and bond type.
            Available options are "connect_the_dots", "van_der_waals", and "hueckel".
        covalent_radius_factor : float
            A scaling factor that is applied to the covalent radii of the atoms when determining
            the bonds with the van-der-Waals method.
        allow_charged_fragments : bool, optional
            If ``True``, fragments with a net charge are allowed when determining the bonds of the
            molecule, by default ``True``.
        embed_chiral : bool, optional
            If ``True``, chiral centers are embedded when determining the bonds of the molecule,
            by default ``True``.

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(
            f"'{self._namespace}' | {self._loc}() | START\n"
            f"> 'connectivity_method':      {connectivity_method}\n"
            f"> 'covalent_radius_factor':   {covalent_radius_factor}\n"
            f"> 'allow_charged_fragments':  {allow_charged_fragments}\n"
            f"> 'embed_chiral':             {embed_chiral}\n"
            "-----"
        )

        # Pre-checks
        self._check_is_initialized(error_message="determining bonds")

        self._check_is_2D("All bonds are already defined.")

        if self.mol_vault.bonds_determined is True:
            _errmsg = (
                "The bonds of the molecule in the molecule vault are already determined "
                "and cannot be determined again."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check input types
        self._check_is_of_type(
            expected_type=str, value=connectivity_method, parameter_name="connectivity_method"
        )
        self._check_is_of_type(
            expected_type=[int, float],
            value=covalent_radius_factor,
            parameter_name="covalent_radius_factor",
        )
        self._check_is_of_type(
            expected_type=bool,
            value=allow_charged_fragments,
            parameter_name="allow_charged_fragments",
        )
        self._check_is_of_type(
            expected_type=bool, value=embed_chiral, parameter_name="embed_chiral"
        )

        # Check covalent_radius_factor input
        covalent_radius_factor = float(covalent_radius_factor)
        if covalent_radius_factor <= 0:
            _errmsg = (
                f"Invalid input to 'covalent_radius_factor': must be > 0 but "
                f"obtained {covalent_radius_factor}."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check connectivity_method input
        connectivity_method = self._check_is_str_in_list(
            parameter_name="connectivity_method",
            value=connectivity_method,
            allowed_values=DETERMINE_BONDS_METHODS,
        )

        # Check if charge is set if Hueckel method is selected
        if connectivity_method == "hueckel" and self.mol_vault.charge is None:
            _errmsg = (
                "Set the charge of the molecule vault with the set_charge() method "
                "before determining bonds with the Hueckel method."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Determine the bonds
        self._determine_bonds(
            connectivity_method=connectivity_method,
            covalent_radius_factor=covalent_radius_factor,
            allow_charged_fragments=allow_charged_fragments,
            embed_chiral=embed_chiral,
        )

        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

    def calculate_electronic_structure(
        self,
        engine: str,
        redox: str = "n",
        prune_by_energy: Optional[Tuple[Union[int, float], str]] = None,
    ) -> None:
        """Calculate the electronic structure of all conformers of a molecule vault hosting
        a 3D molecule.

        The calculation can be performed with either the Psi4 or xtb engine. The ``redox``
        parameter allows to select for which redox states the electronic structure should be
        calculated.

        Parameters
        ----------
        engine : str
            The name of the electronic structure program to be used, either "psi4" or "xtb".
        redox : str, optional
            The redox state for which the electronic structure should be calculated. Can either be

            * "n" (only the actual molecule is calculated),
            * "n-1" (the actual molecule and its one-electron-oxidized form are calculated),
            * "n+1" (the actual molecule and its one-electron-reduced form are calculated), or
            * "all" (the actual molecule and both, its one-electron-reduced and -oxidized form are
              calculated), by default "n".
        prune_by_energy : Optional[Tuple[Union[int, float], str]], optional
            If a value other than ``None`` is provided, all conformers with a relative energy above
            this value are set to be invalid and ignored during feature calculation and any
            further processing. The input must be a 2-tuple in which the first entry is the
            relative energy cutoff value and the second entry is the respective energy unit.
            Supported units are "Eh", "kcal/mol", and "kJ/mol". If ``None``, no pruning is
            performed, by default ``None``.

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(
            f"'{self._namespace}' | {self._loc}() | START\n"
            f"> 'engine':           {engine}\n"
            f"> 'redox':            {redox}\n"
            f"> 'prune_by_energy':  {prune_by_energy}\n"
            "-----"
        )

        # Pre-checks
        self._check_is_initialized(error_message="calculating electronic structures")

        self._check_is_2D(
            "Electronic structure calculations are not feasible for 2D ensembles. "
            "Please provide a 3D ensemble."
        )

        # Check input types
        self._check_is_of_type(expected_type=str, value=engine, parameter_name="engine")
        self._check_is_of_type(expected_type=str, value=redox, parameter_name="redox")

        # The input to prune_by_energy is checked in MolVault.prune_ensemble_by_energy()

        # Check if charge was set
        if self.mol_vault.charge is None:
            _errmsg = (
                "Set the charge of the molecule vault with the set_charge() method "
                "before calculating the electronic structure."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check if multiplicity was set
        if self.mol_vault.multiplicity is None:
            _errmsg = (
                "Set the multiplicity of the molecule vault with the set_multiplicity() "
                "method before calculating the electronic structure."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check engine input
        engine = self._check_is_str_in_list(
            parameter_name="engine", value=engine, allowed_values=ELECTRONIC_STRUCTURE_ENGINES
        )

        # Check redox input
        redox = self._check_is_str_in_list(
            parameter_name="redox", value=redox, allowed_values=REDOX_STATES
        )

        # Format redox input
        if redox == "n":
            states_to_calculate = ["n"]
        elif redox == "n+1":
            states_to_calculate = ["n", "n+1"]
        elif redox == "n-1":
            states_to_calculate = ["n", "n-1"]
        else:
            states_to_calculate = ["n", "n-1", "n+1"]

        # Loop over all states to calculate electronic structure for all conformers
        for state in states_to_calculate:
            # Check if data is already present (either energies or electronic structures)
            _already_present = False
            if state == "n" and any(
                [
                    self.mol_vault.energies_n_read is True,
                    len(self.mol_vault.electronic_strucs_n) > 0,
                ]
            ):
                _already_present = True
            elif state == "n+1" and any(
                [
                    self.mol_vault.energies_n_plus1_read is True,
                    len(self.mol_vault.electronic_strucs_n_plus1) > 0,
                ]
            ):
                _already_present = True
            elif state == "n-1" and any(
                [
                    self.mol_vault.energies_n_minus1_read is True,
                    len(self.mol_vault.electronic_strucs_n_minus1) > 0,
                ]
            ):
                _already_present = True

            if _already_present is True:
                logging.warning(
                    f"'{self._namespace}' | {self._loc}()\nElectronic structure or energy data "
                    f"for state '{state}' is already attached to the molecule vault. Electronic "
                    "structure calculations are skipped for this state."
                )
                continue

            self._calculate_electronic_structure(engine=engine, state=state)

            # Prune conformer ensemble if requested
            if prune_by_energy is not None and state == "n":
                self.mol_vault.prune_ensemble_by_energy(
                    energy_cutoff=prune_by_energy, _called_from=self._loc
                )
            elif prune_by_energy is not None and state != "n":
                logging.warning(
                    f"'{self._namespace}' | {self._loc}()\nPruning the conformer ensemble based on "
                    f"energy values for state '{state}' is not possible. No pruning was performed."
                )

        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

    def featurize_atoms(
        self,
        atom_indices: Union[str, int, List[int]],
        feature_indices: Union[str, int, List[int]],
    ) -> None:
        """Calculate one or multiple features for selected or all atoms.

        A list of all available atom features can be obtained with the ``list_atom_features()``
        method. For certain features, 3D information, electronic structure data or information on
        the chemical bonds in the molecule is required.

        Parameters
        ----------
        atom_indices : Union[str, int, List[int]]
            The indices of the atoms to be featurized. Can be a single index, a list of indices, or
            "all" to consider all atoms.
        feature_indices : Union[str, int, List[int]]
            The indices of the features to be calculated. Can be a single index, a list of indices,
            or "all" to consider all atom features.

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(
            f"'{self._namespace}' | {self._loc}() | START\n"
            f"> 'atom_indices':     {atom_indices}\n"
            f"> 'feature_indices':  {feature_indices}\n"
            "-----"
        )

        # Pre-checks
        self._check_is_initialized(error_message="featurizing atoms")

        if len(self._feature_info) == 0:
            _errmsg = (
                "The feature information file was not sucessfully loaded. Therefore, atom "
                "features cannot be calculated."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise RuntimeError(f"{self._loc}(): {_errmsg}")

        # Check feature indices
        feature_indices = self._check_feature_indices(
            feature_indices=feature_indices,
            feature_type="atom",
            dimensionality=str(self.mol_vault.dimensionality),
        )

        # Reformat feature indices for certain iterable options; atom_indices is set to "all"
        # in certain cases because the 'interable option base feature' must be available for
        # all atoms in order to calculate the actual feature for the selected atoms.
        feature_indices, set_atom_indices_to_all = self._rearrange_feature_indices(
            feature_indices=feature_indices
        )
        if set_atom_indices_to_all is True:
            atom_indices = "all"
            logging.warning(
                f"'{self._namespace}' | {self._loc}()\nThe 'atom_indices' parameter was set to "
                "'all' because at least one of the selected features requires its iterable option "
                "base feature(s) to be available for all atoms."
            )

        # Check atom indices
        atom_indices = self._check_atom_indices(atom_indices=atom_indices)

        logging.info(
            f"'{self._namespace}' | {self._loc}()\nConsidered features: "
            f"{[(idx, self._feature_info[idx]['name']) for idx in feature_indices]}."
        )

        # Calculate features
        logging.info(f"'{self._namespace}' | {self._loc}()\nCalculating atom features now ...")
        self._run_featurization(feature_indices=feature_indices, atom_bond_indices=atom_indices)

        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

    def featurize_bonds(
        self,
        bond_indices: Union[str, int, List[int]],
        feature_indices: Union[str, int, List[int]],
    ) -> None:
        """Calculate one or multiple features for selected or all bonds.

        A list of all available bond features can be obtained with the ``list_bond_features()``
        method. For all bond features, information on the chemical bonds in the molecule is
        required. Some bond features further require 3D information or electronic structure data.

        Parameters
        ----------
        bond_indices : Union[str, int, List[int]]
            The indices of the bonds to be featurized. Can be a single index, a list of indices, or
            "all" to consider all bonds.
        feature_indices : Union[str, int, List[int]]
            The indices of the features to be calculated. Can be a single index, a list of indices,
            or "all" to consider all bond features.

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(
            f"'{self._namespace}' | {self._loc}() | START\n"
            f"> 'bond_indices':     {bond_indices}\n"
            f"> 'feature_indices':  {feature_indices}\n"
            "-----"
        )

        # Pre-checks
        self._check_is_initialized(error_message="featurizing bonds")

        if len(self._feature_info) == 0:
            _errmsg = (
                "The feature information file was not sucessfully loaded. Therefore, bond "
                "features cannot be calculated."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise RuntimeError(f"{self._loc}(): {_errmsg}")

        # Check bond indices
        bond_indices = self._check_bond_indices(bond_indices=bond_indices)

        # Check feature indices
        feature_indices = self._check_feature_indices(
            feature_indices=feature_indices,
            feature_type="bond",
            dimensionality=str(self.mol_vault.dimensionality),
        )
        logging.info(
            f"'{self._namespace}' | {self._loc}()\nConsidered features: "
            f"{[(idx, self._feature_info[idx]['name']) for idx in feature_indices]}."
        )

        # Calculate features
        logging.info(f"'{self._namespace}' | {self._loc}()\nCalculating bond features now ...")
        self._run_featurization(feature_indices=feature_indices, atom_bond_indices=bond_indices)

        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")

    def return_atom_features(
        self,
        atom_indices: Union[str, int, List[int]] = "all",
        output_format: str = "df",
        reduce: bool = False,
        temperature: Union[int, float] = 298.15,
        ignore_invalid: bool = True,
    ) -> Union[pd.DataFrame, Dict[int, Dict[str, Any]], List[Chem.rdchem.Mol], Chem.rdchem.Mol]:
        """Return the calculated atom features after feature calculation.

        The features of selected or all atoms can be returned as a pandas DataFrame, a hierarchical
        dictionary, or as one or multiple RDKit molecule objects with the features embedded as
        atom properties.

        If a dictionary is requested as output format, the outer dictionary keys correspond to the
        atom indices. The values are dictionaries in which the keys are the feature names and the
        values are the respective feature values.

        Parameters
        ----------
        atom_indices : Union[str, int, List[int]], optional
            The indices of the atoms for which features should be returned. If features are
            requested for atoms for which no data was calculated, the feature value will be
            ``NaN``. The input to ``atom_indices`` can be a single index, a list of indices, or
            "all" to consider all atoms, by default "all".
        output_format : str, optional
            The name of the desired output format, can be "df", "dict", or "mol_object". If "df" is
            selected, a pandas DataFrame is returned. If "dict" is selected, the features are
            returned as a hierarchical dictionary. If "mol_object" is selected, one or multiple
            RDKit molecule objects with the features embedded as atom properties are returned, by
            default "df".
        reduce : bool, optional
            This is only relevant for molecule vaults hosting a 3D molecule with more than one
            conformer. If ``True``, the features are reduced to a single value per atom across all
            conformers reporting the minimum, maximum, and mean value for each feature. In
            addition, if energy data is available in the molecule vault, the Boltzmann-weighted
            average value at the provided temperature is reported as well as the data for the
            lowest- and highest-energy conformer. If ``False``, the features are returned for each
            conformer separately, by default ``False``.
        temperature : Union[int, float], optional
            The temperature in Kelvin at which the Boltzmann-weighted values are calculated, by
            default 298.15.
        ignore_invalid : bool, optional
            If set to ``True``, the presence of any invalid conformer in the molecule vault will
            be ignored during feature reduction. If is set to ``False``, the presence of any
            invalid conformer will lead to returning the unreduced features. Note that in both
            cases, invalid conformers are ignored when calculating the mean, min, and max feature
            values.

        Returns
        -------
        Union[pd.DataFrame, Dict[int, Dict[str, Any]], List[Chem.rdchem.Mol], Chem.rdchem.Mol]
            The atom features in the desired output format.
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        return self._return_features(
            feature_type="atom",
            atom_bond_indices=atom_indices,
            output_format=output_format,
            reduce=reduce,
            temperature=temperature,
            ignore_invalid=ignore_invalid,
        )

    def return_bond_features(
        self,
        bond_indices: Union[str, int, List[int]] = "all",
        output_format: str = "df",
        reduce: bool = False,
        temperature: Union[int, float] = 298.15,
        ignore_invalid: bool = True,
    ) -> Union[pd.DataFrame, Dict[int, Dict[str, Any]], List[Chem.rdchem.Mol], Chem.rdchem.Mol]:
        """Return the calculated bond features after feature calculation.

        The features of selected or all bonds can be returned as a pandas DataFrame, a hierarchical
        dictionary, or as one or multiple RDKit molecule objects with the features embedded as
        bond properties.

        If a dictionary is requested as output format, the outer dictionary keys correspond to the
        bond indices. The values are dictionaries in which the keys are the feature names and the
        values are the respective feature values.

        Parameters
        ----------
        bond_indices : Union[str, int, List[int]], optional
            The indices of the bonds for which features should be returned. If features are
            requested for bonds for which no data was calculated, the feature value will be
            ``NaN``. The input to ``bond_indices`` can be a single index, a list of indices, or
            "all" to consider all bonds, by default "all".
        output_format : str, optional
            The name of the desired output format, can be "df", "dict", or "mol_object". If "df" is
            selected, a pandas DataFrame is returned. If "dict" is selected, the features are
            returned as a hierarchical dictionary. If "mol_object" is selected, one or multiple
            RDKit molecule objects with the features embedded as bond properties are returned, by
            default "df".
        reduce : bool, optional
            This is only relevant for molecule vaults hosting a 3D molecule with more than one
            conformer. If ``True``, the features are reduced to a single value per bond across all
            conformers reporting the minimum, maximum, and mean value for each feature. In
            addition, if energy data is available in the molecule vault, the Boltzmann-weighted
            average value at the provided temperature is reported as well as the data for the
            lowest- and highest-energy conformer. If ``False``, the features are returned for each
            conformer separately, by default ``False``.
        temperature : Union[int, float], optional
            The temperature in Kelvin at which the Boltzmann-weighted values are calculated, by
            default 298.15.
        ignore_invalid : bool, optional
            If set to ``True``, the presence of any invalid conformer in the molecule vault will
            be ignored during feature reduction. If is set to ``False``, the presence of any
            invalid conformer will lead to returning the unreduced features. Note that in both
            cases, invalid conformers are ignored when calculating the mean, min, and max feature
            values.

        Returns
        -------
        Union[pd.DataFrame, Dict[int, Dict[str, Any]], List[Chem.rdchem.Mol], Chem.rdchem.Mol]
            The bond features in the desired output format.
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        return self._return_features(
            feature_type="bond",
            atom_bond_indices=bond_indices,
            output_format=output_format,
            reduce=reduce,
            temperature=temperature,
            ignore_invalid=ignore_invalid,
        )

    def clear_atom_feature_cache(self, origin: Optional[Union[str, List[str]]] = None) -> None:
        """Clear the atom feature cache of the molecule vault.

        This method can be used to clear previously calculated atom features from the feature
        cache of the molecule vault to recalculate them (e.g., after changing the configuration
        settings of a featurizer, see the ``set_options()`` method).

        Parameters
        ----------
        origin : Optional[Union[str, List[str]]]
            The name or a list of the names of the program(s) of the feature(s) to be cleared
            (e.g., "rdkit", "xtb"), by default ``None``. If ``None``, all features are cleared.

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        self._clear_feature_cache(feature_type="atom", origin=origin)

    def clear_bond_feature_cache(self, origin: Optional[Union[str, List[str]]] = None) -> None:
        """Clear the bond feature cache of the molecule vault.

        This method can be used to clear previously calculated bond features from the feature
        cache of the molecule vault to recalculate them (e.g., after changing the configuration
        settings of a featurizer, see the ``set_options()`` method).

        Parameters
        ----------
        origin : Optional[Union[str, List[str]]]
            The name or a list of the names of the program(s) of the feature(s) to be cleared
            (e.g., "rdkit", "xtb"), by default ``None``. If ``None``, all features are cleared.

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        self._clear_feature_cache(feature_type="bond", origin=origin)

    def add_custom_featurizer(self, custom_metadata: Dict[str, Any]) -> None:
        """Add a custom featurizer to the BONAFIDE framework.

        After successfully calling this method, the custom feature is assigned its own feature
        index and can be used like any other built-in feature.

        Parameters
        ----------
        custom_metadata : Dict[str, Any]
            A dictionary containing the required information on the custom featurizer. It must
            contain the following data:

            * name (str): The name of the custom feature.
            * origin (str): The origin program of the custom feature (e.g., "custom")
            * feature_type (str): The type of the custom feature (either "atom" or "bond").
            * dimensionality (str): The dimensionality of the custom feature (either "2D" or "3D").
            * data_type (str): The data type of the custom feature specified as string (either
              "str", "int", "float", or "bool").
            * requires_electronic_structure_data (bool): Whether electronic structure data is
              required for calculating the custom feature.
            * requires_bond_data (bool): Whether bond data is required for calculating the
              custom feature.
            * requires_charge (bool): Whether the charge of the molecule is required for
              calculating the custom feature.
            * requires_multiplicity (bool): Whether the multiplicity of the molecule is
              required for calculating the custom feature.
            * config_path (dict): Dictionary of optional parameters passed to the custom
              featurizer. The keys of this dictionary will be available as attributes in the
              custom featurizer class.
            * factory (callable): The factory class for calculating the custom feature. It
              must inherit from ``BaseFeaturizer`` from ``bonafide/utils/base_featurizer.py``.

        Returns
        -------
        None
        """
        self._loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.info(
            f"'{self._namespace}' | {self._loc}() | START\n"
            f"> 'custom_metadata':  {custom_metadata}\n"
            "-----"
        )

        # Pre-checks
        if len(self._feature_info) == 0:
            _errmsg = (
                "The feature information file was not sucessfully loaded. Therefore, a custom "
                "featurizer cannot be added."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise RuntimeError(f"{self._loc}(): {_errmsg}")

        if len(self._feature_config) == 0:
            _errmsg = (
                "The default feature configuration settings file was not sucessfully loaded. "
                "Therefore, a custom featurizer cannot be added."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise RuntimeError(f"{self._loc}(): {_errmsg}")

        # Check input type
        self._check_is_of_type(
            expected_type=dict, value=custom_metadata, parameter_name="custom_metadata"
        )

        # Validate user input
        origin, custom_metadata = custom_featurizer_data_validator(
            custom_metadata=custom_metadata,
            feature_info=self._feature_info,
            namespace=str(self._namespace),
            loc=self._loc,
            feature_config=self._feature_config,
        )

        # Add new feature to the FeatureFactory dictionary
        factory_name = custom_metadata["factory"].__name__
        FEATURE_FACTORIES[factory_name] = custom_metadata["factory"]

        # Change the data in factory to the actually needed information
        custom_metadata["factory"] = factory_name

        # Add new feature to the feature config dictionary
        self._feature_config[origin] = custom_metadata["config_path"]
        self._check_config_dict()

        # Add new feature to self._feature_info
        feature_idx = max(self._feature_info.keys()) + 1
        custom_metadata["config_path"] = (
            origin  # Change the data in config_path to the actually needed information
        )
        self._feature_info[feature_idx] = custom_metadata
        self._process_feature_info_dict()

        logging.info(
            f"'{self._namespace}' | {self._loc}()\nCustom {custom_metadata['feature_type']} "
            f"feature '{custom_metadata['name']}' (produced in '{factory_name}') successfully "
            f"added to the feature space with INDEX {feature_idx}."
        )

        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")
