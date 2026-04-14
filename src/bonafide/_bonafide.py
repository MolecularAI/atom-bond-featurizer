"""BONAFIDE base class with all private methods."""

from __future__ import annotations

import copy
import json
import logging
import os
import tomllib
from abc import ABC, abstractmethod
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from bonafide._bonafide_utils import _AtomBondFeaturizerUtils
from bonafide.utils.constants import (
    ATTRIBUTE_BLACK_LIST,
    ELECTRONIC_STRUCTURE_DATA_FILE_EXTENSIONS,
    FEATURE_TYPES,
    OUTPUT_TYPES,
)
from bonafide.utils.feature_factories import FEATURE_FACTORIES
from bonafide.utils.feature_output import FeatureOutput
from bonafide.utils.helper_functions import clean_up, flatten_dict, standardize_string
from bonafide.utils.helper_functions_chemistry import bind_smiles_with_xyz, get_molecular_formula
from bonafide.utils.input_validation import config_data_validator
from bonafide.utils.io_ import extract_energy_from_string, read_smiles
from bonafide.utils.logging_format import IndentationFormatter
from bonafide.utils.sp_psi4 import Psi4SP
from bonafide.utils.sp_xtb import XtbSP

if TYPE_CHECKING:
    import ipywidgets
    from mendeleev import element
    from PIL import PngImagePlugin

    from bonafide.utils.molecule_vault import MolVault


class _AtomBondFeaturizer(ABC, _AtomBondFeaturizerUtils):
    _atom_feature_indices_2D: List[int]
    _atom_feature_indices_3D: List[int]
    _bond_feature_indices_2D: List[int]
    _bond_feature_indices_3D: List[int]
    _feature_config: Dict[str, Any]
    _feature_info: Dict[int, Dict[str, Any]]
    _feature_info_df: pd.DataFrame
    _functional_groups_smarts: Dict[str, List[Tuple[str, Chem.rdchem.Mol]]]
    _init_directory: str
    _keep_output_files: bool
    _loc: str
    _namespace: Optional[str]
    _output_directory: Optional[str]
    _periodic_table: Dict[str, element]
    mol_vault: MolVault

    @abstractmethod
    def list_atom_features(self, **kwargs: Any) -> pd.DataFrame: ...

    @abstractmethod
    def list_bond_features(self, **kwargs: Any) -> pd.DataFrame: ...

    @abstractmethod
    def print_options(self, origin: Optional[Union[str, List[str]]]) -> None: ...

    @abstractmethod
    def set_options(self, configs: Union[Tuple[str, Any], List[Tuple[str, Any]]]) -> None: ...

    @abstractmethod
    def read_input(
        self,
        input_value: Union[str, Chem.rdchem.Mol],
        namespace: str,
        input_format: str,
        read_energy: bool,
        prune_by_energy: Optional[Tuple[Union[int, float], str]],
        output_directory: Optional[str],
    ) -> None: ...

    @abstractmethod
    def show_molecule(
        self,
        index_type: Optional[str],
        in_3D: bool,
        image_size: Tuple[int, int],
    ) -> Union[PngImagePlugin.PngImageFile, ipywidgets.VBox]: ...

    @abstractmethod
    def set_charge(self, charge: int) -> None: ...

    @abstractmethod
    def set_multiplicity(self, multiplicity: int) -> None: ...

    @abstractmethod
    def attach_smiles(
        self,
        smiles: str,
        align: bool,
        connectivity_method: str,
        covalent_radius_factor: Union[int, float],
    ) -> None: ...

    @abstractmethod
    def attach_electronic_structure(
        self, electronic_structure_data: Union[str, List[str]], state: str
    ) -> None: ...

    @abstractmethod
    def determine_bonds(
        self,
        connectivity_method: str,
        covalent_radius_factor: Union[int, float],
        allow_charged_fragments: bool,
        embed_chiral: bool,
    ) -> None: ...

    @abstractmethod
    def calculate_electronic_structure(
        self,
        engine: str,
        redox: str,
        prune_by_energy: Optional[Tuple[Union[int, float], str]],
    ) -> None: ...

    @abstractmethod
    def featurize_atoms(
        self,
        atom_indices: Union[str, int, List[int]],
        feature_indices: Union[str, int, List[int]],
    ) -> None: ...

    @abstractmethod
    def featurize_bonds(
        self,
        bond_indices: Union[str, int, List[int]],
        feature_indices: Union[str, int, List[int]],
    ) -> None: ...

    @abstractmethod
    def return_atom_features(
        self,
        atom_indices: Union[str, int, List[int]],
        output_format: str,
        reduce: bool,
        temperature: Union[int, float],
        ignore_invalid: bool,
    ) -> Union[pd.DataFrame, Dict[int, Dict[str, Any]], List[Chem.rdchem.Mol], Chem.rdchem.Mol]: ...

    @abstractmethod
    def return_bond_features(
        self,
        bond_indices: Union[str, int, List[int]],
        output_format: str,
        reduce: bool,
        temperature: Union[int, float],
        ignore_invalid: bool,
    ) -> Union[pd.DataFrame, Dict[int, Dict[str, Any]], List[Chem.rdchem.Mol], Chem.rdchem.Mol]: ...

    @abstractmethod
    def add_custom_featurizer(self, custom_metadata: Dict[str, Any]) -> None: ...

    def _init_logging(self, log_file_name: Any) -> None:
        """Set up the logging to a file with the provided log file name.

        Initially, the input is checked for validity. If the input is valid, the logging is set up.

        Parameters
        ----------
        log_file_name : Any
            The name of the log file to which the logging messages should be written.

        Returns
        -------
        None
        """
        # Check input type
        self._check_is_of_type(
            expected_type=str, value=log_file_name, parameter_name="log_file_name"
        )

        # Check if input is empty
        if log_file_name.strip() == "":
            _errmsg = "Invalid input to 'log_file_name': must not be an empty string."
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check if log file already exists
        if os.path.exists(log_file_name):
            _errmsg = (
                f"The log file at '{os.path.abspath(log_file_name)}' already exists. "
                "Remove or rename the file before running BONAFIDE with the provided "
                "log file name."
            )
            raise FileExistsError(f"{self._loc}(): {_errmsg}")

        # Remove potential old handlers to avoid logging conflicts
        _root_logger = logging.getLogger()
        for handler in _root_logger.handlers[:]:  # Use slice to avoid modification during iteration
            # Don't remove pytest-related handlers when tests are run
            if (
                type(handler).__name__ in ["LogCaptureHandler", "_LiveLoggingNullHandler"]
                and os.environ.get("BONAFIDE_PYTEST_TESTING_SESSION", "0") == "1"
            ):
                continue
            handler.close()
            _root_logger.removeHandler(hdlr=handler)

        # Setup logging
        _handler = logging.FileHandler(filename=log_file_name)
        _format = "%(asctime)s | %(levelname)s | %(message)s"
        _dateformat = "%Y-%m-%d %H:%M:%S"

        _handler.setFormatter(IndentationFormatter(fmt=_format, datefmt=_dateformat))
        logging.basicConfig(level=logging.INFO, handlers=[_handler])

        # Capture warnings from other programs that calculate the features
        logging.captureWarnings(capture=True)

        # Log file header
        logging.info(
            "=======================================================================\n"
            r"            ____   __   __ _   __   ____  ___  ____  ____             " + "\n"
            r"           (  _ \ /  \ (  ( \ / _\ (  __)(   )(    \(  __)            " + "\n"
            r"            ) B ((  O )/  N // A  \ )F_)  )I(  ) D ( )E_)             " + "\n"
            r"           (____/ \__/ \_)__)\_/\_/(__)  (___)(____/(____)            " + "\n"
            "                                                                       \n"
            "               Features for Atoms and Bonds in Molecules               \n"
            "=======================================================================\n"
            f"* Version:                 {version(distribution_name='bonafide')}\n"
            "* Documentation:           https://molecularai.github.io/atom-bond-featurizer\n"
            "* GitHub:                  https://github.com/MolecularAI/atom-bond-featurizer\n"
            f"* DOI:                     10.26434/chemrxiv.15001386/v1\n"
            f"* Reference:               ChemRXiv 2026\n"
            "\n"
            f"* Installation directory:  {os.path.dirname(__file__)}\n"
        )

    def _load_config_file(self) -> None:
        """Load the ``_feature_config.toml`` configuration file that stores the default setting
        parameters for the individual featurization programs.

        After reading the file, it is checked for disallowed keys that would interfere with
        the rest of the code.

        Returns
        -------
        None
        """
        # Read the file
        _file_name = "_feature_config.toml"
        _toml_config_file_path = os.path.join(os.path.dirname(__file__), _file_name)
        try:
            with open(_toml_config_file_path, "rb") as config_file:
                self._feature_config = tomllib.load(config_file)
        except Exception as e:
            _errmsg = f"Error while reading the '{_file_name}' file: {e}."
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise RuntimeError(f"{self._loc}(): {_errmsg}")

        # Check the read data
        self._check_config_dict()

        logging.info(
            f"'{self._namespace}' | {self._loc}()\nDefault feature configuration settings "
            f"successfully loaded from '{_toml_config_file_path}'."
        )

    def _check_config_dict(self) -> None:
        """Check for disallowed keys in the configuration settings dictionary.

        The keys listed in ``ATTRIBUTE_BLACK_LIST`` are not allowed in the configuration settings
        dictionary because they are used internally for other data.

        Returns
        -------
        None
        """
        # Check for disallowed keys in the toml file that would interfere with the rest of the code
        all_config_keys = flatten_dict(self._feature_config, [])
        for key in all_config_keys:
            if key in ATTRIBUTE_BLACK_LIST:
                _errmsg = (
                    f"'{key}' is not an allowed key in the '_feature_config.toml' "
                    "configuration file."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

    def _load_feature_info_file(self) -> None:
        """Read the ``_feature_info.json`` feature configuration file that stores all implemented
        features with their associated metadata.

        After reading the file, it is processed to define the atom and bond feature indices
        for 2D and 3D molecules.

        Returns
        -------
        None
        """
        # Read file
        _file_name = "_feature_info.json"
        _feature_info_file_path = os.path.join(os.path.dirname(__file__), _file_name)
        try:
            with open(_feature_info_file_path, "r") as feature_file:
                self._feature_info = json.load(feature_file)
        except Exception as e:
            _errmsg = f"Error while reading the '{_file_name}' file: {e}."
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise RuntimeError(f"{self._loc}(): {_errmsg}")

        self._feature_info = {int(idx): data for idx, data in self._feature_info.items()}
        self._process_feature_info_dict()

        logging.info(
            f"'{self._namespace}' | {self._loc}()\nFeature information file successfully loaded "
            f"from '{_feature_info_file_path}'.\n"
        )

    def _process_feature_info_dict(self) -> None:
        """Process the feature information dictionary to define the atom and bond feature for 2D
        and 3D molecules and set up the feature information pandas DataFrame.

        All 2D features are also valid for 3D molecules.

        Returns
        -------
        None
        """
        df = pd.DataFrame(self._feature_info).T
        df = df.reset_index(names="INDEX")
        df = df.set_index("INDEX")
        df.index = pd.Index(df.index.map(int), dtype=object)

        # 2D atom features
        self._atom_feature_indices_2D = list(
            df[(df["feature_type"] == "atom") & (df["dimensionality"] == "2D")].index
        )

        # 2D bond features
        self._bond_feature_indices_2D = list(
            df[(df["feature_type"] == "bond") & (df["dimensionality"] == "2D")].index
        )

        # 3D atom features
        self._atom_feature_indices_3D = list(df[df["feature_type"] == "atom"].index)

        # 3D bond features
        self._bond_feature_indices_3D = list(df[df["feature_type"] == "bond"].index)

        self._feature_info_df = df

    def _list_features(self, feature_type: str, **kwargs: Any) -> pd.DataFrame:
        """Display all available features for atoms or bonds.

        Parameters
        ----------
        feature_type : str
            The type of features to be listed, either "atom" or "bond".
        **kwargs: Any
            Additional optional keyword arguments for filtering the feature DataFrame. If empty,
            all features are returned.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the selected features and their characteristics.
        """
        logging.info(
            f"'{self._namespace}' | {self._loc}() | START\n> 'arguments':  {kwargs}\n-----"
        )

        # Pre-checks
        if self._feature_info_df.shape == (0, 0):
            _errmsg = (
                "The feature information file was not successfully loaded. Therefore, "
                f"{feature_type} features cannot be listed."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise RuntimeError(f"{self._loc}(): {_errmsg}")

        df = self._feature_info_df.copy()
        df["name"] = [n.split("-")[-1] for n in df["name"]]
        df = df[df["feature_type"] == feature_type]

        _cols = list(df.columns)
        _cols.remove("feature_type")
        _filters = []

        for col_name, filter_key in kwargs.items():
            # Check if valid column name was passed
            if col_name not in _cols:
                _errmsg = (
                    f"Invalid input to '**kwargs': '{col_name}' is not a valid column in "
                    f"the feature DataFrame, available: {_cols}."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

            # Format filter key dependent on passed column name
            if col_name in ["name", "origin", "feature_type", "data_type", "config_path"]:
                filter_key = standardize_string(inp_data=filter_key)
            elif col_name in ["dimensionality"]:
                filter_key = standardize_string(inp_data=filter_key, case="upper")

            # Filter DataFrame
            df = df[df[col_name] == filter_key]
            _filters.append(f"{col_name}={filter_key}")

        logging.info(
            f"'{self._namespace}' | {self._loc}()\nFeature DataFrame was compiled. "
            f"Applied filters: {_filters}."
        )
        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")
        return df

    def _set_options(self, config_path: str, value: Any) -> None:
        """Execute the change of the configuration settings for the individual programs used for
        feature calculation.

        Parameters
        ----------
        config_path : str
            The path to the configuration setting to be changed (point-separated).
        value : Any
            The new value for the configuration setting.

        Returns
        -------
        None
        """
        # Walk through the configurations dictionary
        config_key_list = config_path.split(".")
        section = self._feature_config
        for key in config_key_list[:-1]:
            try:
                section = section[key]
            except KeyError:
                _errmsg = (
                    f"Invalid input to 'configs': '{key}' is not a valid branch within "
                    "the configuration settings tree."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

        # Set the value
        setting_name = config_key_list[-1]
        if setting_name not in section:
            _errmsg = (
                f"Invalid input to 'configs': '{setting_name}' is not a valid option "
                f"for '{'.'.join(config_key_list[:-1])}'."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        _old_value = section[setting_name]
        section[setting_name] = value

        # Validate the new configuration settings
        params = self._get_configs(key_list=config_key_list[:-1], include_root_data=True)
        params["feature_info"] = self._feature_info

        try:
            params = config_data_validator(
                config_path=config_key_list[:-1], params=params, _namespace=self._namespace
            )
        except Exception as e:
            section[setting_name] = _old_value
            raise e
        else:
            logging.info(
                f"'{self._namespace}' | {self._loc}()\n'{'.'.join(config_key_list)}' was updated. "
                f"The new value is: {params[config_key_list[-1]]}."
            )

    def _setup_output_directory(self, dir_path: str) -> None:
        """Create a folder for all output files created during feature calculation.

        Parameters
        ----------
        dir_path : str
            The path to the output directory to be created.

        Returns
        -------
        None
        """
        dir_path = os.path.abspath(dir_path)

        # Check if provided output directory already exists
        if os.path.isdir(dir_path):
            _errmsg = (
                f"The directory at '{dir_path}' already exists and can therefore not be used as "
                "output directory."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise FileExistsError(f"{self._loc}(): {_errmsg}")

        # Check if provided output directory is writable
        _parent_dir = os.path.dirname(dir_path)
        if not os.access(_parent_dir, os.W_OK):
            _errmsg = (
                f"The directory at '{_parent_dir}' is not writable and can therefore not be used "
                "as parent directory of the output directory."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise PermissionError(f"{self._loc}(): {_errmsg}")

        # Create the output directory
        try:
            os.mkdir(path=dir_path)
        except Exception as e:
            _errmsg = f"Creating the output directory at '{dir_path}' failed: {e}."
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise RuntimeError(f"{self._loc}(): {_errmsg}")

        # Keep output files when the user specified an output directory and internally save the
        # output directory path
        self._keep_output_files = True
        self._output_directory = os.path.abspath(dir_path)
        logging.info(
            f"'{self._namespace}' | {self._loc}()\nOutput directory was created at "
            f"'{os.path.abspath(dir_path)}'. All generated output files will be kept."
        )

    def _attach_smiles(
        self, smiles: str, align: bool, connectivity_method: str, covalent_radius_factor: float
    ) -> None:
        """Execute the attachment of a SMILES string to a molecule vault hosting a 3D molecule.

        For details on how atom connectivity is determined in the SMILES attachment process,
        please refer to the RDKit documentation
        (https://rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html, last accessed on
        29.09.2025).

        Parameters
        ----------
        smiles : str
            The SMILES string that should be attached to the molecule vault.
        align : bool, optional
            If ``True``, the atom indices of the initially provided 3D structure(s) are preserved,
            if ``False``, the atoms are re-ordered according to the order in the SMILES string.
        connectivity_method : str
            The name of the method that is used to determine atom connectivity when binding the
            SMILES string to the molecule vault. Available options are "connect_the_dots",
            "van_der_waals", and "hueckel".
        covalent_radius_factor : float
            A scaling factor that is applied to the covalent radii of the atoms when determining the
            atom connectivity with the van-der-Waals method.

        Returns
        -------
        None
        """
        # Get the first mol object of the molecule vault as reference
        ref_mol = Chem.Mol(self.mol_vault.mol_objects[0])

        # Read in the provided SMILES string
        smiles_mol, error_message = read_smiles(smiles)
        if error_message is not None:
            logging.error(f"'{self._namespace}' | {self._loc}()\n{error_message}")
            raise ValueError(f"{self._loc}(): {error_message}")

        # Check if provided SMILES is compatible with already existing molecule in the vault
        assert smiles_mol is not None  # for type checker
        _ref_atom_count = ref_mol.GetNumAtoms()
        _smiles_atom_count = smiles_mol.GetNumAtoms()
        if _ref_atom_count != _smiles_atom_count:
            _errmsg = (
                "The number of atoms of the structure(s) in the molecule vault "
                f"({_ref_atom_count}) does not match the number of atoms from the SMILES string "
                f"({_smiles_atom_count}). Did you add hydrogen atoms to the SMILES string?"
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Double-check with molecular formula
        _ref_formula = get_molecular_formula(ref_mol)
        _smiles_formula = get_molecular_formula(smiles_mol)
        if _ref_formula != _smiles_formula:
            _errmsg = (
                f"Molecular formula of the molecule in the molecule vault ({_ref_formula}) does "
                "not match the molecular formula of the molecule represented by the SMILES string "
                f"({_smiles_formula})."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Add smiles string to the molecule vault
        self.mol_vault.smiles = smiles

        if align is False:
            _wmsg = (
                "The align parameter is set to False. This will change the atom order of the "
                "molecule in the vault to the one defined by the SMILES string. Ensure that the "
                "new atom order is consistent with any follow-up steps; most importantly, with the "
                "atom order of electronic structure data files that might be attached and used for "
                "feature calculation."
            )
            logging.warning(f"'{self._namespace}' | {self._loc}()\n{_wmsg}")

        # Update the bond information of the mol objects of the molecule vault
        for idx, mol in enumerate(self.mol_vault.mol_objects):
            # Copy smiles_mol to avoid modifying it throughout the loop
            _smiles_mol = Chem.Mol(smiles_mol)

            # Try to attach the SMILES string to the conformer
            try:
                new_mol, error_message = bind_smiles_with_xyz(
                    smiles_mol=_smiles_mol,
                    xyz_mol=mol,
                    align=align,
                    connectivity_method=connectivity_method,
                    covalent_radius_factor=covalent_radius_factor,
                    charge=self.mol_vault.charge,
                )

            except Exception as e:
                _errmsg = (
                    f"Attaching the SMILES string to the conformer with index {idx} failed: "
                    f"{e}. Therefore, the conformer is set to be invalid."
                )
                self.mol_vault.is_valid[idx] = False
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")

            else:
                # Do attachment
                if error_message is None:
                    self.mol_vault.mol_objects[idx] = new_mol
                    _infmsg = (
                        f"'{self._namespace}' | {self._loc}()\nSMILES string was "
                        f"successfully attached to the conformer with index {idx}.\n"
                    )
                    if align is True:
                        _infmsg += "Initial atom order was maintained within the molecule vault."
                    else:
                        _infmsg += "Atom order of the provided SMILES string was applied."
                    logging.info(_infmsg)
                # Error handling
                else:
                    _errmsg = (
                        f"Attaching the SMILES string to the conformer with index {idx} failed: "
                        f"{error_message}. Therefore, fhe conformer is set to be invalid."
                    )
                    self.mol_vault.is_valid[idx] = False
                    logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")

        # Set bonds as determined
        self.mol_vault.bonds_determined = True

        # Update elements list with the potentially changed atom order
        self.mol_vault.get_elements()

    def _attach_energy(
        self,
        energy_data: List[Tuple[Union[int, float], str]],
        state: str,
    ) -> None:
        """Execute the attachment of energy data to a molecule vault hosting a 3D molecule.

        Parameters
        ----------
        energy_data : List[Tuple[Union[int, float], str]]
            The list of 2-tuples containing the energy values and respective units to be attached to
            the molecule vault.
        state : str
            The redox state of the energy data to be attached. Can either be "n" (actual molecule),
            "n+1" (actual molecule plus one electron), or "n-1" (actual molecule minus one
            electron).

        Returns
        -------
        None
        """
        # Check if energy was already read
        if any(
            [
                state == "n" and self.mol_vault.energies_n_read is True,
                state == "n-1" and self.mol_vault.energies_n_minus1_read is True,
                state == "n+1" and self.mol_vault.energies_n_plus1_read is True,
            ]
        ):
            _errmsg = (
                f"Energy data for state '{state}' is already attached to the molecule vault "
                "and cannot be attached again."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Get correct energy data list
        if state == "n":
            _energy_data_list = self.mol_vault.energies_n
            _energy_data_list_as_submitted = self.mol_vault._input_energies_n

        if state == "n-1":
            _energy_data_list = self.mol_vault.energies_n_minus1
            _energy_data_list_as_submitted = self.mol_vault._input_energies_n_minus1

        if state == "n+1":
            _energy_data_list = self.mol_vault.energies_n_plus1
            _energy_data_list_as_submitted = self.mol_vault._input_energies_n_plus1

        # Process all energy data
        energy_value: Optional[Union[int, float]]
        for idx, (energy_value, unit) in enumerate(energy_data):
            # Check energy value
            if energy_value is None:
                pass
            else:
                self._check_is_of_type(
                    expected_type=[int, float],
                    value=energy_value,
                    parameter_name="energy_data",
                    prefix=f"energy value for conformer with index {idx}",
                )

            # Check unit
            self._check_is_of_type(
                expected_type=str,
                value=unit,
                parameter_name="energy_data",
                prefix=f"energy unit for conformer with index {idx}",
            )

            # Handle energy input (check unit and convert to kJ/mol)
            if energy_value is not None:
                _energy_value_pair = f"{energy_value} {unit}"
                try:
                    energy_value_as_submitted, unit_as_submitted, energy_value, error_message = (
                        extract_energy_from_string(line=_energy_value_pair)
                    )
                except Exception as e:
                    _errmsg = f"Reading of input to 'energy_data' failed: {e}."
                    logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                    raise RuntimeError(f"{self._loc}(): {_errmsg}")

                if error_message is not None:
                    logging.error(f"'{self._namespace}' | {self._loc}()\n{error_message}")
                    raise ValueError(f"{self._loc}(): {error_message}.")

            else:
                energy_value_as_submitted = None
                unit_as_submitted = unit

            # Attach data
            _energy_data_list.append((energy_value, "kj_mol"))
            _energy_data_list_as_submitted.append((energy_value_as_submitted, unit_as_submitted))

            if state == "n":
                self.mol_vault.energies_n_read = True
            if state == "n-1":
                self.mol_vault.energies_n_minus1_read = True
            if state == "n+1":
                self.mol_vault.energies_n_plus1_read = True

            if energy_value is None:
                logging.warning(
                    f"'{self._namespace}' | {self._loc}()\nThe energy of conformer with index "
                    f"{idx} for state '{state}' is None. The conformer is therefore set to be "
                    "invalid. This is probably not intended."
                )
                self.mol_vault.is_valid[idx] = False

            logging.info(
                f"'{self._namespace}' | {self._loc}()\nEnergy was attached to conformer with "
                f"index {idx} for state '{state}'."
            )

    def _attach_electronic_structure(
        self,
        electronic_struc_list: List[str],
        _el_struc_list: List[str],
        _el_struc_types: List[str],
        state: str,
    ) -> None:
        """Execute the attachment of electronic structure data file(s) to a molecule vault hosting
        a 3D molecule.

        Parameters
        ----------
        electronic_struc_list : List[str]
            The list of paths to the electronic structure data files to be attached to the
            molecule vault.
        _el_struc_list : List[str]
            The attribute of the ``MolVault`` object that stores the paths to the electronic
            structure data files.
        _el_struc_types : List[str]
            The attribute of the ``MolVault`` object that stores the file types of the
            electronic structure data files (file extensions).
        state : str
            The redox state of the electronic structure data to be attached. Can either be "n"
            (actual molecule), "n+1" (actual molecule plus one electron), or "n-1" (actual molecule
            minus one electron).

        Returns
        -------
        None
        """
        # Loop over all provided electronic structure files and process them
        for idx, el_struc_file in enumerate(electronic_struc_list):
            # Determine file type
            if el_struc_file is not None:
                file_type = os.path.splitext(el_struc_file)[-1][1:]

                # Check if the file exists
                if not os.path.exists(el_struc_file):
                    _errmsg = (
                        f"Invalid input to 'electronic_structure_data': path to the input file at "
                        f"'{el_struc_file}' is invalid."
                    )
                    logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                    raise FileNotFoundError(f"{self._loc}(): {_errmsg}")

                file_path = os.path.join(os.getcwd(), el_struc_file)

                logging.info(
                    f"'{self._namespace}' | {self._loc}()\nElectronic structure data "
                    f"(*.{file_type} file) was attached to conformer with index {idx} for "
                    f"state '{state}'.",
                )

                # Check file type
                _file_type = standardize_string(inp_data=file_type)
                if _file_type not in ELECTRONIC_STRUCTURE_DATA_FILE_EXTENSIONS:
                    logging.warning(
                        f"'{self._namespace}' | {self._loc}()\nElectronic structure data files of "
                        f"type '{file_type}' may lead to unexpected behavior or errors during "
                        "feature calculation. Ensure that the obtained results are valid. BONAFIDE "
                        "was developed and extensively tested with 'molden' and 'fchk' files."
                    )

            else:
                file_path = None
                file_type = None
                logging.warning(
                    f"'{self._namespace}' | {self._loc}()\nElectronic structure data of type None "
                    f"was attached to conformer with index {idx} for state '{state}'. The "
                    f"conformer is therefore set to be invalid. This is probably not intended."
                )
                self.mol_vault.is_valid[idx] = False

            # Attach the electronic structure file and the file type to the conformer ensemble
            _el_struc_list.append(file_path)
            _el_struc_types.append(file_type)

    def _determine_bonds(
        self,
        connectivity_method: str,
        covalent_radius_factor: float,
        allow_charged_fragments: bool,
        embed_chiral: bool,
    ) -> None:
        """Execute the determination of the chemical bonds of each conformer of a molecule vault
        hosting a 3D molecule.

        For details on how the bonds are determined, please refer to the RDKit documentation
        (https://rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html, last accessed on
        29.09.2025).

        Parameters
        ----------
        connectivity_method : str
            The name of the method that is used to determine the bonds. Available options are
            "connect_the_dots", "van_der_waals", and "hueckel".
        covalent_radius_factor : float
            A scaling factor that is applied to the covalent radii of the atoms when determining the
            bonds with the van-der-Waals method.
        allow_charged_fragments : bool
            If ``True``, charged fragments are allowed when determining the bonds. If ``False``,
            unpaired electrons are introduced according to the valence of the respective atom.
        embed_chiral : bool
            If ``True``, chiral information will be added to the molecule when determining the
            bonds.

        Returns
        -------
        None
        """
        # Set bonds as determined
        self.mol_vault.bonds_determined = True

        # Prepare input to RDKit (which method to use)
        _use_hueckel = False
        _use_vdw = True
        if connectivity_method == "hueckel":
            _use_hueckel = True
        if connectivity_method == "connect_the_dots":
            _use_vdw = False

        # Determine bonds for each conformer
        for idx, mol in enumerate(self.mol_vault.mol_objects):
            try:
                if _use_hueckel is True:
                    rdDetermineBonds.DetermineBonds(
                        mol=mol,
                        useHueckel=_use_hueckel,
                        charge=self.mol_vault.charge,
                        allowChargedFragments=allow_charged_fragments,
                        embedChiral=embed_chiral,
                    )
                else:
                    rdDetermineBonds.DetermineBonds(
                        mol=mol,
                        covFactor=covalent_radius_factor,
                        allowChargedFragments=allow_charged_fragments,
                        embedChiral=embed_chiral,
                        useVdw=_use_vdw,
                    )

            except Exception as e:
                _errmsg = (
                    "Determining the chemical bonds failed for conformer with index "
                    f"{idx} which is therefore set to be invalid: {e}."
                )
                self.mol_vault.is_valid[idx] = False
                logging.warning(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            else:
                logging.info(
                    f"'{self._namespace}' | {self._loc}()\nChemical bonds were determined for "
                    f"conformer with index {idx}."
                )

        # Ensure that the conformers are identical
        self.mol_vault.compare_conformers()

        # Clean up properties after determining bonds to remove undesired values
        self.mol_vault.clean_properties()

        # Clean up (after Hueckel calculation)
        clean_up(to_be_removed=["nul", "run.out"])

    def _calculate_electronic_structure(self, engine: str, state: str) -> None:
        """Execute the calculation of the electronic structure of all conformers of a molecule
        vault hosting a 3D molecule.

        Parameters
        ----------
        engine : str
            The name of the electronic structure program to be used, either "psi4" or "xtb".
        state : str
            The redox state of the electronic structure data to be calculated. Can either be "n"
            (actual molecule), "n+1" (actual molecule plus one electron), or "n-1" (actual molecule
            minus one electron).

        Returns
        -------
        None
        """
        # Fetch program-specific parameters and check their validity
        params = self._feature_config[engine]
        params = config_data_validator(
            config_path=[engine], params=params, _namespace=self._namespace
        )

        params["mol_vault"] = self.mol_vault
        params["state"] = state
        params["_keep_output_files"] = self._keep_output_files

        # Adjust charge and multiplicity if needed
        _init_charge = self.mol_vault.charge
        _init_multiplicity = self.mol_vault.multiplicity

        try:
            assert isinstance(self.mol_vault.charge, int)  # for type checker
            assert isinstance(self.mol_vault.multiplicity, int)  # for type checker
            if state == "n+1":
                self.mol_vault.charge -= 1
            if state == "n-1":
                self.mol_vault.charge += 1
            if state in ["n+1", "n-1"]:
                if self.mol_vault.multiplicity == 1:
                    self.mol_vault.multiplicity = 2
                else:
                    self.mol_vault.multiplicity -= 1

            # Initialize respective class for sp calculation(s)
            sp: Union[Psi4SP, XtbSP]
            if engine == "psi4":
                sp = Psi4SP(**params)
            if engine == "xtb":
                sp = XtbSP(**params)

            # Change current working directory to the output files directory
            assert self._output_directory is not None  # for type checker
            os.chdir(self._output_directory)

            try:
                # Run the calculation(s)
                _write_el_struc_file = False
                if self._keep_output_files is True:
                    _write_el_struc_file = True
                energies, electronic_strucs = sp.run(
                    state=state, write_el_struc_file=_write_el_struc_file
                )
            finally:
                # Always reset current working directory to the path where the featurizer was initialized
                os.chdir(self._init_directory)

            # Save the results to the molecule vault
            _init_log = self._loc
            logging.info(
                f"'{self._namespace}' | {self._loc}()\nSingle-point energy calculations done for "
                f"state '{state}'. The calculated energy data for all conformers is automatically "
                f"attached to the molecule vault for state '{state}'."
            )
            self.attach_energy(energy_data=energies, state=state)  # type: ignore[attr-defined]
            self._loc = _init_log

            # Automatically attach electronic structure to the molecule vault
            if self._keep_output_files is True:
                logging.info(
                    f"'{self._namespace}' | {self._loc}()\nThe calculated electronic structure "
                    f"data files are automatically attached to the molecule vault for state '{state}'."
                )
                self.attach_electronic_structure(
                    electronic_structure_data=electronic_strucs,  # type: ignore[arg-type]
                    state=state,
                )
            else:
                logging.warning(
                    f"'{self._namespace}' | {self._loc}()\nThe electronic structure data files were "
                    "calculated but not attached to the molecule vault because the output files were "
                    "deleted after the calculations. Specify an output directory in the read_input() "
                    "method ('output_directory' parameter) to keep the output files and automatically "
                    "attach them to the molecule vault."
                )

        finally:
            # Always reset charge and multiplicity to initial values
            self.mol_vault.charge = _init_charge
            self.mol_vault.multiplicity = _init_multiplicity

    def _run_featurization(self, feature_indices: List[int], atom_bond_indices: List[int]) -> None:
        """Calculate the requested atom or bond features.

        Features are calculated by running through four nested loops in the following order:

        1. Loop over all requested feature indices.
        2. Loop over all iterable options (if applicable, otherwise a dummy iterable option None
           is used that remains without any effect).
        3. Loop over all conformers in the molecule vault.
        4. Loop over all requested atom or bond indices.

        Parameters
        ----------
        feature_indices : List[int]
            The indices of the features to be calculated.
        atom_bond_indices : List[int]
            The indices of the atoms or bonds for which the features should be calculated.

        Returns
        -------
        None
        """
        elements = copy.deepcopy(self.mol_vault.elements)
        charge = copy.deepcopy(self.mol_vault.charge)
        multiplicity = copy.deepcopy(self.mol_vault.multiplicity)

        #################################
        # Loop over all feature indices #
        #################################
        for feature_idx in feature_indices:
            f_configs = self._feature_info[feature_idx]
            feature_name = f_configs["name"]
            feature_type = f_configs["feature_type"]
            feature_dimensionality = f_configs["dimensionality"]
            config_key_list = f_configs["config_path"].split(".")
            data_type = f_configs["data_type"]
            factory = f_configs["factory"]

            requires_el_struc_data = f_configs["requires_electronic_structure_data"]
            requires_bond_data = f_configs["requires_bond_data"]
            requires_charge = f_configs["requires_charge"]
            requires_multiplicity = f_configs["requires_multiplicity"]

            # Check if for the requested feature, data on the electronic structure is required
            assert isinstance(self.mol_vault.size, int)  # for type checker
            if all(
                [
                    requires_el_struc_data is True,
                    len(self.mol_vault.electronic_strucs_n) < self.mol_vault.size,
                ]
            ):
                _errmsg = (
                    f"For calculating the '{feature_name}' feature (INDEX = {feature_idx}), "
                    "electronic structure data is required but is not available. Attach "
                    "precomputed electronic structure data with the attach_electronic_structure() "
                    "method or calculate it from scratch with the calculate_electronic_structure() "
                    "method."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

            # Check if for the requested feature, data on the bonds is required
            if requires_bond_data is True and self.mol_vault.bonds_determined is False:
                _errmsg = (
                    f"For calculating the '{feature_name}' feature (INDEX = {feature_idx}), "
                    "bond data is required but is not available. Determine bonds with the "
                    "determine_bonds() method or provide bond information, e.g., through a "
                    "SMILES string or an SD file."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

            # Check if for the requested feature, the charge of the molecule is required
            if requires_charge is True and self.mol_vault.charge is None:
                _errmsg = (
                    f"For calculating the '{feature_name}' feature (INDEX = {feature_idx}), "
                    "the charge of the molecule is required but is not set. Set the charge with "
                    "the set_charge() method."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

            # Check if for the requested feature, the multiplicity of the molecule is required
            if requires_multiplicity is True and self.mol_vault.multiplicity is None:
                _errmsg = (
                    f"For calculating the '{feature_name}' feature (INDEX = {feature_idx}), "
                    "the multiplicity of the molecule is required but is not set. Set the "
                    "multiplicity with the set_multiplicity() method before calculating "
                    "this feature."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

            # Check if a 3D feature is attempted to be calculated for a 2D ensemble
            # Not required because _atom_feature_indices_2D and _bond_feature_indices_2D exclude
            # all 3D features.

            # Set up the input dictionary for the feature calculation and validate the data
            params = self._get_configs(key_list=config_key_list, include_root_data=True)

            # Temporarily add the entire self._feature_info to the params dictionary,
            # this is removed by config_data_validator after validation. This is required to
            # validate the input to iterable_options for the autocorrelation features (see
            # _ValidateIterableIntOptionMixin in utils/input_validation.py)
            params["feature_info"] = self._feature_info
            params = config_data_validator(
                config_path=config_key_list, params=params, _namespace=self._namespace
            )

            # Add additional information to the params dictionary
            params["_periodic_table"] = self._periodic_table
            params["_functional_groups_smarts"] = self._functional_groups_smarts
            params["_keep_output_files"] = self._keep_output_files
            params["feature_name"] = feature_name
            params["feature_type"] = feature_type
            params["feature_dimensionality"] = feature_dimensionality
            params["elements"] = elements
            params["charge"] = charge
            params["multiplicity"] = multiplicity
            params["ensemble_dimensionality"] = self.mol_vault.dimensionality

            # Add feature caches to the params dictionary
            if feature_type == "atom":
                params["feature_cache"] = self.mol_vault.atom_feature_cache_n
                params["feature_cache_n_minus1"] = self.mol_vault.atom_feature_cache_n_minus1
                params["feature_cache_n_plus1"] = self.mol_vault.atom_feature_cache_n_plus1
            if feature_type == "bond":
                params["feature_cache"] = self.mol_vault.bond_feature_cache

            params["global_feature_cache"] = self.mol_vault.global_feature_cache

            # Handle iterable option. In case no iterable option is present, a dummy iterable
            # option is created (None) which remains without any effect
            if "iterable_option" not in params:
                _iterable_option = [None]
            else:
                _iterable_option = params["iterable_option"]

            ##############################
            # Loop over iterable options #
            ##############################
            for iter_opt in _iterable_option:
                params["iterable_option"] = iter_opt

                ############################
                # Loop over all conformers #
                ############################
                for conf_idx, mol in enumerate(self.mol_vault.mol_objects):
                    # Skip conformers that were labeled as invalid
                    if self.mol_vault.is_valid[conf_idx] is False:
                        logging.warning(
                            f"'{self._namespace}' | {self._loc}()\nSkipping conformer with index "
                            f"{conf_idx} because it was set to be invalid."
                        )
                        continue

                    conformer_name = self.mol_vault.conformer_names[conf_idx]

                    coordinates = None
                    if self.mol_vault.dimensionality == "3D":
                        coordinates = mol.GetConformer(0).GetPositions()

                    energy_n = None
                    if len(self.mol_vault.energies_n) > 0:
                        energy_n = self.mol_vault.energies_n[conf_idx]

                    energy_n_minus1 = None
                    if len(self.mol_vault.energies_n_minus1) > 0:
                        energy_n_minus1 = self.mol_vault.energies_n_minus1[conf_idx]

                    energy_n_plus1 = None
                    if len(self.mol_vault.energies_n_plus1) > 0:
                        energy_n_plus1 = self.mol_vault.energies_n_plus1[conf_idx]

                    electronic_struc_n = None
                    electronic_struc_type_n = None
                    if len(self.mol_vault.electronic_strucs_n) > 0:
                        electronic_struc_n = self.mol_vault.electronic_strucs_n[conf_idx]
                        electronic_struc_type_n = self.mol_vault.electronic_struc_types_n[conf_idx]

                    electronic_struc_n_plus1 = None
                    if len(self.mol_vault.electronic_strucs_n_plus1) > 0:
                        electronic_struc_n_plus1 = self.mol_vault.electronic_strucs_n_plus1[
                            conf_idx
                        ]

                    electronic_struc_n_minus1 = None
                    if len(self.mol_vault.electronic_strucs_n_minus1) > 0:
                        electronic_struc_n_minus1 = self.mol_vault.electronic_strucs_n_minus1[
                            conf_idx
                        ]

                    params["mol"] = Chem.Mol(mol)  # protect the actual mol object by copying it
                    params["conformer_name"] = conformer_name
                    params["conformer_idx"] = conf_idx
                    params["coordinates"] = coordinates
                    params["energy_n"] = energy_n
                    params["energy_n_minus1"] = energy_n_minus1
                    params["energy_n_plus1"] = energy_n_plus1
                    params["electronic_struc_n"] = electronic_struc_n
                    params["electronic_struc_n_plus1"] = electronic_struc_n_plus1
                    params["electronic_struc_n_minus1"] = electronic_struc_n_minus1
                    params["electronic_struc_type_n"] = electronic_struc_type_n

                    ######################################
                    # Loop over all atom or bond indices #
                    ######################################
                    for atom_bond_idx in atom_bond_indices:
                        params["atom_bond_idx"] = atom_bond_idx

                        # Change current working directory to the output files directory
                        assert self._output_directory is not None  # for type checker
                        os.chdir(self._output_directory)

                        try:
                            # Calculate feature with callable factory class
                            calc_feature = FEATURE_FACTORIES[factory]()
                            feature_value, error_message = calc_feature(**params)
                        finally:
                            # Always reset current working directory to path where featurizer was initialized
                            os.chdir(self._init_directory)

                        # Adjust feature name in case the feature class modified it
                        # (when iterable options are used)
                        _feature_name = calc_feature.feature_name

                        # Write the feature value to the RDKit mol object
                        self._set_feature(
                            conf_idx=conf_idx,
                            mol=mol,
                            atom_bond_idx=atom_bond_idx,
                            feature_type=feature_type,
                            feature_name=_feature_name,
                            feature_value=feature_value,
                            error_message=error_message,
                            data_type=data_type,
                        )

        # Clean up the mol objects after property calculation
        self.mol_vault.clean_properties()

    def _set_feature(
        self,
        conf_idx: int,
        mol: Chem.rdchem.Mol,
        atom_bond_idx: int,
        feature_type: str,
        feature_name: str,
        feature_value: Optional[Union[int, float, bool, str]],
        error_message: Optional[str],
        data_type: str,
    ) -> None:
        """Set a feature value for the specified atom or bond.

        The feature is stored as property of the respective RDKit atom or bond object.

        Parameters
        ----------
        conf_idx : int
            The index of the conformer in the molecule vault.
        mol : Chem.rdchem.Mol
            The RDKit molecule object within which the feature value should be set.
        atom_bond_idx : int
            The index of the atom or bond for which the feature value should be set.
        feature_type : str
            The type of the feature, either "atom" or "bond".
        feature_name : str
            The name of the feature for which the value should be set.
        feature_value : Optional[Union[int, float, bool, str]]
            The calculated feature value that should be set. If the feature calculation failed,
            this is ``None``.
        error_message : Optional[str]
            Any error message that occurred during feature calculation. If no error occurred, this
            is ``None``.
        data_type : str
            The expected data type of the feature value, either int, float, bool, or str.

        Returns
        -------
        None
        """
        # Log error message if not None
        if error_message is not None:
            logging.error(
                f"'{self._namespace}' | {self._loc}()\nFeature calculation for '{feature_name}' "
                f"of feature type '{feature_type}' (index = {atom_bond_idx}) and data type "
                f"'{data_type}' failed for conformer with index {conf_idx}: {error_message}."
            )
            return

        # Check if feature value is None
        if feature_value is None:
            logging.warning(
                f"'{self._namespace}' | {self._loc}()\nFeature calculation for '{feature_name}' "
                f"of feature type '{feature_type}' (index = {atom_bond_idx}) and data type "
                f"'{data_type}' terminated without error but the feature value is None for "
                f"conformer with index {conf_idx}."
            )
            return

        # Log if the feature calculation run successfully and the feature value is '_inaccessible'.
        # This is for example the case for certain vdW surface features if the atom does not
        # contribute to the vdW surface.
        _helper_bool = False
        try:
            _helper_bool = feature_value == "_inaccessible"
        except Exception:
            pass
        if _helper_bool is True and error_message is None:
            logging.warning(
                f"'{self._namespace}' | {self._loc}()\nFeature calculation for '{feature_name}' "
                f"of feature type '{feature_type}' (index = {atom_bond_idx}) and data type "
                f"'{data_type}' terminated without error but the feature value is '_inaccessible' "
                f"for conformer with index {conf_idx}. Probably the feature is not defined for "
                f"the requested {feature_type}.",
            )
            data_type = "str"

        # Check if correct data type is provided
        _obtained_data_type = type(feature_value).__name__
        if _obtained_data_type != data_type:
            _errmsg = (
                f"Data type '{_obtained_data_type}' of feature value '{feature_value}' associated "
                f"with feature '{feature_name}' calculated for {feature_type} with index "
                f"'{atom_bond_idx}' does not match the expected data type '{data_type}'."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise RuntimeError(f"{self._loc}(): {_errmsg}")

        # Get atom or bond object
        feature_type = self._check_is_str_in_list(
            parameter_name="feature_type",
            value=feature_type,
            allowed_values=FEATURE_TYPES,
        )
        if feature_type == "atom":
            obj = mol.GetAtomWithIdx(atom_bond_idx)
        if feature_type == "bond":
            obj = mol.GetBondWithIdx(atom_bond_idx)

        # Save result
        if data_type == "int":
            obj.SetIntProp(feature_name, feature_value)
        elif data_type == "float":
            obj.SetDoubleProp(feature_name, feature_value)
        elif data_type == "bool":
            obj.SetBoolProp(feature_name, feature_value)
        elif data_type == "str":
            obj.SetProp(feature_name, feature_value)
        else:
            _errmsg = (
                f"Data type {data_type} associated with '{feature_name}' is not supported. "
                "Supported feature types are int, float, bool, and str."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise RuntimeError(f"{self._loc}(): {_errmsg}")

        if feature_value != "_inaccessible":
            logging.info(
                f"'{self._namespace}' | {self._loc}()\nFeature calculation for '{feature_name}' "
                f"of feature type '{feature_type}' (index = {atom_bond_idx}) and data type "
                f"'{data_type}' successful for conformer with index {conf_idx}."
            )

    def _return_features(
        self,
        feature_type: str,
        atom_bond_indices: Union[str, int, List[int]],
        output_format: str,
        reduce: bool,
        temperature: Union[int, float],
        ignore_invalid: bool,
    ) -> Union[pd.DataFrame, Dict[int, Dict[str, Any]], List[Chem.rdchem.Mol], Chem.rdchem.Mol]:
        """Return the calculated atom or bond features.

        Parameters
        ----------
        feature_type : str
            The type of features to be returned, either "atom" or "bond".
        atom_bond_indices : Union[str, int, List[int]], optional
            The indices of the atoms or bonds for which features should be returned.
        output_format : str, optional
            The name of the desired output format, can be "df", "dict", or "mol_object".
        reduce : bool, optional
            If ``True``, the features are reduced to a set of single values per atom or bond
            across all conformers. If ``False``, the features are returned for each conformer
            separately.
        temperature : Union[int, float], optional
            The temperature in Kelvin at which the Boltzmann-weighted values are calculated.
        ignore_invalid : bool, optional
            Whether to ignore conformers that were labeled as invalid when calculating the
            features.

        Returns
        -------
        Union[pd.DataFrame, Dict[int, Dict[str, Any]], List[Chem.rdchem.Mol], Chem.rdchem.Mol]
            The atom or bond features in the desired output format.
        """
        _loggingmsg = (
            f"'{self._namespace}' | {self._loc}() | START\n"
            f"> 'feature_type':        {feature_type}\n"
            f"> '{feature_type}_indices':        {atom_bond_indices}\n"
            f"> 'output_format':       {output_format}\n"
            f"> 'reduce':              {reduce}\n"
        )

        # Try because reduce input not yet type-checked
        try:
            if reduce is True:
                _loggingmsg += (
                    f"> 'temperature':         {temperature}\n"
                    f"> 'ignore_invalid':      {ignore_invalid}\n"
                )
        except Exception:
            pass
        _loggingmsg += "-----"
        logging.info(_loggingmsg)

        # Pre-checks
        self._check_is_initialized(error_message="returning features")

        # Check input types
        self._check_is_of_type(
            expected_type=str, value=output_format, parameter_name="output_format"
        )
        self._check_is_of_type(expected_type=bool, value=reduce, parameter_name="reduce")
        self._check_is_of_type(
            expected_type=[int, float], value=temperature, parameter_name="temperature"
        )
        self._check_is_of_type(
            expected_type=bool, value=ignore_invalid, parameter_name="ignore_invalid"
        )

        # Check provided temperature
        if temperature <= 0:
            _errmsg = (
                "Invalid input to 'temperature': must be greater than 0 but "
                f"obtained {temperature}."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check provided output_format
        output_format = self._check_is_str_in_list(
            parameter_name="output_format", value=output_format, allowed_values=OUTPUT_TYPES
        )

        # Check provided indices and if features were calculated
        _data_found = False
        if feature_type == "atom":
            atom_bond_indices_checked = self._check_atom_indices(atom_indices=atom_bond_indices)

            for mol in self.mol_vault.mol_objects:
                for atom in mol.GetAtoms():
                    if atom.GetPropsAsDict() != {}:
                        _data_found = True
                        break

        if feature_type == "bond":
            atom_bond_indices_checked = self._check_bond_indices(bond_indices=atom_bond_indices)

            for mol in self.mol_vault.mol_objects:
                for bond in mol.GetBonds():
                    if bond.GetPropsAsDict() != {}:
                        _data_found = True
                        break

        # Check if any features were calculated
        if _data_found is False:
            _errmsg = (
                f"No '{feature_type}' features were calculated yet. Run the "
                f"featurize_{feature_type}s() method before trying to return '{feature_type}' "
                "features."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        logging.info(
            f"'{self._namespace}' | {self._loc}()\nFetching '{feature_type}' features from "
            "molecule vault."
        )

        # Update the Boltzmann weights before getting the features (if required)
        assert isinstance(self.mol_vault.size, int)  # for type checker
        if all(
            [
                self.mol_vault.dimensionality == "3D",
                self.mol_vault.energies_n_read is True,
                self.mol_vault.size > 1,
                reduce is True,
            ]
        ):
            self.mol_vault.update_boltzmann_weights(
                temperature=temperature,
                ignore_invalid=ignore_invalid,
            )

        # Get features
        fout = FeatureOutput(
            mol_vault=self.mol_vault,
            indices=atom_bond_indices_checked,
            feature_type=feature_type,
            reduce=reduce,
            ignore_invalid=ignore_invalid,
            _loc=self._loc,
        )
        res = fout.get_results(output_format=output_format)

        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")
        return res

    def _clear_feature_cache(
        self, feature_type: str, origin: Optional[Union[str, List[str]]]
    ) -> None:
        """Clear the atom or bond feature cache of the molecule vault.

        Parameters
        ----------
        feature_type : str
            The type of the feature(s) to be cleared, either "atom" or "bond".
        origin : Optional[Union[str, List[str]]]
            The name or a list of the names of the program(s) of the feature(s) to be cleared
            (e.g., "rdkit", "xtb"). If ``None``, all features of the specified type are cleared.

        Returns
        -------
        None
        """
        logging.info(
            f"'{self._namespace}' | {self._loc}() | START\n"
            f"> 'origin':        {origin}\n"
            f"> 'feature_type':  {feature_type}\n"
            "-----"
        )

        # Check input types
        _inpt = type(origin)
        if origin is not None:
            _errmsg = (
                "Invalid input to 'origin': must be either None, of type str, or a list of "
                f"strings but obtained {_inpt.__name__}."
            )
            if _inpt == str:
                origin = [standardize_string(inp_data=origin)]
            elif _inpt == list:
                for o in origin:
                    if type(o) != str:
                        logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                        raise TypeError(f"{self._loc}(): {_errmsg}")
                origin = [standardize_string(inp_data=o) for o in origin]
            else:
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise TypeError(f"{self._loc}(): {_errmsg}")

        # Check if provided origins are valid
        if origin is not None:
            _valid_origins = list(self._feature_config.keys())
            for o in origin:
                self._check_is_str_in_list(
                    parameter_name="origin", value=o, allowed_values=_valid_origins
                )

        # Clear feature cache
        self.mol_vault.clear_feature_cache_(feature_type=feature_type, origins=origin)

        # Also reset _functional_groups_smarts to allow the addition of new functional groups
        self._functional_groups_smarts = {}

        logging.info(f"'{self._namespace}' | {self._loc}() | DONE\n")
