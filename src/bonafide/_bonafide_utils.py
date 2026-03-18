"""Utility methods for BONAFIDE."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

from rdkit import Chem

from bonafide.utils.helper_functions import standardize_string

if TYPE_CHECKING:
    from bonafide.utils.molecule_vault import MolVault


class _AtomBondFeaturizerUtils:
    """Mixin class providing utility methods for BONAFIDE."""

    _atom_feature_indices_2D: List[int]
    _atom_feature_indices_3D: List[int]
    _bond_feature_indices_2D: List[int]
    _bond_feature_indices_3D: List[int]
    _feature_config: Dict[str, Any]
    _feature_info: Dict[int, Dict[str, Any]]
    _loc: str
    _namespace: Optional[str]
    dimensionality: Optional[str]
    mol_vault: Optional[MolVault]

    def _check_is_initialized(self, error_message: str) -> None:
        """Check if the molecule vault is initialized.

        Parameters
        ----------
        error_message : str
            A string that is added to the final error message that is raised if the molecule
            vault is not initialized.

        Returns
        -------
        None
        """
        if self.mol_vault is None:
            _errmsg = (
                "Read in a SMILES string or an input file with a single or an ensemble of "
                f"conformers before {error_message}."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

    def _check_is_2D(self, error_message: str) -> None:
        """Check if the molecule vault is of dimensionality "2D".

        Parameters
        ----------
        error_message : str
            A string that is added to the final error message that is raised if the molecule
            vault is of dimensionality "2D".

        Returns
        -------
        None
        """
        assert self.mol_vault is not None  # for type checker
        if self.mol_vault.dimensionality == "2D":
            _errmsg = f"The initialized molecule vault is of dimensionality '2D'. {error_message}"
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

    def _check_is_of_type(
        self,
        expected_type: Union[Any, List[Any]],
        value: Any,
        parameter_name: str,
        prefix: str = "",
    ) -> None:
        """Check if a provided value is of a specific type.

        Parameters
        ----------
        expected_type : Union[Any, List[Any]]
            The expected type(s) of the provided value; multiple types can be tolerated.
        value : Any
            The value to be checked.
        parameter_name : str
            The name of the parameter that is checked.
        prefix : str, optional
            An optional prefix that is added to the error message, by default "".

        Returns
        -------
        None
        """
        type_dict = {
            str: "str",
            bool: "bool",
            int: "int",
            float: "float",
            list: "list",
            dict: "dict",
            tuple: "tuple",
            None: "None",
            Chem.rdchem.Mol: "rdkit.Chem.rdchem.Mol",
        }

        if type(expected_type) != list:
            expected_type = [expected_type]

        if len(expected_type) == 1:
            insert_str = type_dict[expected_type[0]]
        elif len(expected_type) == 2:
            insert_str = f"{type_dict[expected_type[0]]} or {type_dict[expected_type[1]]}"
        else:
            insert_str = ", ".join([type_dict[t] for t in expected_type[:-1]])
            insert_str += f", or {type_dict[expected_type[-1]]}"

        if prefix == "":
            prefix = "must be of type "
        else:
            prefix += " must be of type "

        if value is None:
            inpt = None
            inpn = "None"
        else:
            inpt = type(value)
            inpn = inpt.__name__

        if inpt not in expected_type:
            _errmsg = (
                f"Invalid input to '{parameter_name}': {prefix}{insert_str} but obtained {inpn}."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise TypeError(f"{self._loc}(): {_errmsg}")

    def _check_is_str_in_list(
        self, parameter_name: str, value: Any, allowed_values: List[Any]
    ) -> str:
        """Check if a provided string is in a list of (allowed) values.

        The provided value is standardized before the check. The allowed values are not
        standardized.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter that is checked.
        value : Any
            The value to be checked.
        allowed_values : List[Any]
            A list of allowed values.

        Returns
        -------
        str
            The standardized input value if it is in the list of allowed values.
        """
        # Standardize the input value
        _value = standardize_string(inp_data=value)

        # Check
        if _value not in allowed_values:
            _errmsg = (
                f"Invalid input to '{parameter_name}': '{value}' is not supported, "
                f"available: {allowed_values}."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        return _value

    def _check_atom_indices(self, atom_indices: Union[str, int, List[int]]) -> List[int]:
        """Check and format atom indices.

        Parameters
        ----------
        atom_indices : Union[str, int, List[int]]
            The indices of the atoms to be processed. Can be a single index, a list of indices, or
            "all" to consider all atoms.

        Returns
        -------
        List[int]
            A list of validated atom indices.
        """
        # Return the full list of atom indices if "all" is requested
        _inpt = type(atom_indices)
        if _inpt == str:
            assert self.mol_vault is not None  # for type checker
            if standardize_string(inp_data=atom_indices) == "all":
                atom_indices = list(range(self.mol_vault.mol_objects[0].GetNumAtoms()))
                return atom_indices
            _errmsg = (
                f"Invalid input to 'atom_indices': '{atom_indices}' is not supported as input. "
                "The only valid string input is 'all' to address all atoms."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check if atom_indices is a single integer or a list of integers
        if isinstance(atom_indices, int) and _inpt == int:
            atom_indices = [atom_indices]
        elif isinstance(atom_indices, list):
            for atom_idx in atom_indices:
                _inpt = type(atom_idx)
                if _inpt != int:
                    _errmsg = (
                        f"Invalid input to 'atom_indices': provided atom index {atom_idx} of type "
                        f"{_inpt.__name__} is of wrong type. All provided atom indices must be of "
                        "type int. To request features for all atoms, pass 'all'."
                    )
                    logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                    raise TypeError(f"{self._loc}(): {_errmsg}")
        else:
            _errmsg = (
                "Invalid input to 'atom_indices': must be either a single integer or a list of "
                "integers. To request features for all atoms, pass 'all'."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise TypeError(f"{self._loc}(): {_errmsg}")

        # Check if any specified index is out of bounds
        _final_idx_list = []
        for atom_idx in atom_indices:
            assert self.mol_vault is not None  # for type checker
            if atom_idx < 0 or atom_idx >= self.mol_vault.mol_objects[0].GetNumAtoms():
                _errmsg = (
                    f"Invalid input to 'atom_indices': provided atom index {atom_idx} is out of "
                    f"bounds. The molecule contains {self.mol_vault.mol_objects[0].GetNumAtoms()} "
                    "atoms. To request features for all atoms, pass 'all'."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

            # Remove duplicates
            if atom_idx not in _final_idx_list:
                _final_idx_list.append(atom_idx)
            else:
                logging.warning(
                    f"'{self._namespace}' | {self._loc}()\nInput to 'atom_indices' contained a "
                    f"duplicate ({atom_idx}), which was removed."
                )

        return _final_idx_list

    def _check_bond_indices(self, bond_indices: Union[str, int, List[int]]) -> List[int]:
        """Check and format bond indices.

        Parameters
        ----------
        bond_indices : Union[str, int, List[int]]
            The indices of the bonds to be processed. Can be a single index, a list of indices, or
            "all" to consider all bonds.

        Returns
        -------
        List[int]
            A list of validated bond indices.
        """
        # Return the full list of bond indices if "all" is requested
        _inpt = type(bond_indices)
        if _inpt == str:
            assert self.mol_vault is not None  # for type checker
            if standardize_string(inp_data=bond_indices) == "all":
                bond_indices = list(range(self.mol_vault.mol_objects[0].GetNumBonds()))
                return bond_indices
            _errmsg = (
                f"Invalid input to 'bond_indices': '{bond_indices}' is not supported as input. "
                "The only valid string input is 'all' to address all bonds."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Check if bond_indices is a single integer or a list of integers
        if isinstance(bond_indices, int) and _inpt == int:
            bond_indices = [bond_indices]
        elif isinstance(bond_indices, list):
            for bond_idx in bond_indices:
                _inpt = type(bond_idx)
                if _inpt != int:
                    _errmsg = (
                        f"Invalid input to 'bond_indices': provided bond index {bond_idx} of type "
                        f"{_inpt.__name__} is of wrong type. All provided bond indices must be of "
                        "type int. To request features for all bonds, pass 'all'."
                    )
                    logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                    raise TypeError(f"{self._loc}(): {_errmsg}")
        else:
            _errmsg = (
                "Invalid input to 'bond_indices': must be either a single integer or a list of "
                "integers. To request features for all bonds, pass 'all'."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise TypeError(f"{self._loc}(): {_errmsg}")

        # Check if any specified index is out of bounds
        _final_idx_list = []
        for bond_idx in bond_indices:
            assert self.mol_vault is not None  # for type checker
            if bond_idx < 0 or bond_idx >= self.mol_vault.mol_objects[0].GetNumBonds():
                _errmsg = (
                    f"Invalid input to 'bond_indices': provided bond index {bond_idx} is out of "
                    f"bounds. The molecule contains {self.mol_vault.mol_objects[0].GetNumBonds()} "
                    "bonds. To request features for all bonds, pass 'all'."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

            # Remove duplicates
            if bond_idx not in _final_idx_list:
                _final_idx_list.append(bond_idx)
            else:
                logging.warning(
                    f"'{self._namespace}' | {self._loc}()\nInput to 'bond_indices' contained a "
                    f"duplicate ({bond_idx}), which was removed."
                )

        return _final_idx_list

    def _check_feature_indices(
        self,
        feature_indices: Union[str, int, List[int]],
        feature_type: str,
        dimensionality: str,
    ) -> List[int]:
        """Check and format feature indices.

        Parameters
        ----------
        feature_indices : Union[str, int, List[int]]
            The indices of the features to be processed. Can be a single index, a list of indices,
            or "all" to consider all features.
        feature_type : str
            The type of the feature, either "atom" or "bond".
        dimensionality : str
            The dimensionality of the molecule vault, either "2D" or "3D".

        Returns
        -------
        List[int]
            A list of validated feature indices.
        """
        # Select the allowed feature indices depending on the feature type and the dimensionality
        # of the molecule vault
        if feature_type == "atom" and dimensionality == "2D":
            allowed_feature_indices = self._atom_feature_indices_2D
        if feature_type == "bond" and dimensionality == "2D":
            allowed_feature_indices = self._bond_feature_indices_2D

        if feature_type == "atom" and dimensionality == "3D":
            allowed_feature_indices = self._atom_feature_indices_3D
        if feature_type == "bond" and dimensionality == "3D":
            allowed_feature_indices = self._bond_feature_indices_3D

        # Check input types
        _inpt = type(feature_indices)
        if _inpt == str:
            if standardize_string(inp_data=feature_indices) == "all":
                return allowed_feature_indices
            else:
                _errmsg = (
                    f"Invalid input to 'feature_indices': '{feature_indices}' is not supported. "
                    "Provided feature indices must be either a single integer or a list of "
                    "integers. Setting 'feature_indices' to 'all' is also allowed to request "
                    "all features."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

        if isinstance(feature_indices, int) and _inpt == int:
            feature_indices = [feature_indices]

        if _inpt not in [int, list]:
            _errmsg = (
                "Invalid input to 'feature_indices': provided feature indices must be either a "
                f"single integer or a list of integers, not of type {_inpt.__name__}. Setting "
                "'feature_indices' to 'all' is also allowed to request all features."
            )
            logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
            raise TypeError(f"{self._loc}(): {_errmsg}")

        _final_idx_list = []
        assert hasattr(feature_indices, "__iter__")  # for type checker
        for feature_idx in feature_indices:
            # Check type of each feature index
            _inpt = type(feature_idx)
            if _inpt != int:
                _errmsg = (
                    f"Invalid input to 'feature_indices': provided feature index '{feature_idx}' "
                    f"of type {_inpt.__name__} is of wrong type. All provided feature indices must "
                    "be of type int. Setting 'feature_indices' to 'all' is also allowed to request "
                    "all features."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise TypeError(f"{self._loc}(): {_errmsg}")

            # Check if feature is allowed
            if feature_idx not in allowed_feature_indices:
                _errmsg = (
                    f"Invalid input to 'feature_indices': provided feature index '{feature_idx}' "
                    f"is not supported for feature type '{feature_type}'. Consult the "
                    "list_atom_features() and list_bond_features() method for finding out about "
                    "allowed feature indices. Setting feature_indices to 'all' is also allowed to "
                    "request all features."
                )
                logging.error(f"'{self._namespace}' | {self._loc}()\n{_errmsg}")
                raise ValueError(f"{self._loc}(): {_errmsg}")

            # Remove duplicates
            if feature_idx not in _final_idx_list:
                _final_idx_list.append(feature_idx)
            else:
                logging.warning(
                    f"'{self._namespace}' | {self._loc}()\nInput to 'feature_indices' contained a "
                    f"duplicate ({feature_idx}), which was removed."
                )

        _final_idx_list_res = cast(List[int], _final_idx_list)  # for type checker
        return _final_idx_list_res

    def _rearrange_feature_indices(self, feature_indices: List[int]) -> Tuple[List[int], bool]:
        """Organize the feature indices list such that the required feature indices for the
        iterable options of the 'atom-autocorrelation' features are at the beginning of the
        feature indices list.

        This is required to ensure that the respective features are computed before the
        'atom-autocorrelation' features are calculated. Moreover, these prerequisite features must
        be computed for all atoms, hence the method also returns a flag that indicates whether the
        atom indices should be set to "all".

        Parameters
        ----------
        feature_indices : List[int]
            The indices of the features to be calculated.

        Returns
        -------
        Tuple[List[int], bool]
            A tuple containing:

            * The rearranged list of feature indices in which the iterable options feature indices
              are at the beginning.
            * A boolean flag that indicates whether the atom indices should be set to "all".
        """
        _final_idx_list = []
        _set_atom_indices_to_all = False

        for feature_idx in feature_indices:
            f_configs = self._feature_info[feature_idx]
            feature_name = f_configs["name"]
            config_key_list = f_configs["config_path"].split(".")
            params = self._get_configs(config_key_list)

            # Handle the iterable options for the autocorrelation features
            if "iterable_option" in params and any(["atom-autocorrelation" in feature_name]):
                _set_atom_indices_to_all = True
                for iter_opt in params["iterable_option"]:
                    if iter_opt not in _final_idx_list:
                        _final_idx_list.append(iter_opt)

        # Add the remaining features
        for feature_idx in feature_indices:
            if feature_idx not in _final_idx_list:
                _final_idx_list.append(feature_idx)

        return _final_idx_list, _set_atom_indices_to_all

    def _get_configs(self, key_list: List[str], include_root_data: bool = False) -> Dict[str, Any]:
        """Extract configuration settings from ``_feature_config``.

        Parameters
        ----------
        key_list : List[str]
            A list of keys that specify the section from which the configuration settings should
            be read.
        include_root_data : bool, optional
            Whether to include root data in the returned configuration settings, by default
            ``False``. If set to ``True``, the lowest-level key value pairs of the specified
            section (based on ``key_list``) are returned together with the actual data.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the configuration settings from the specified section.
        """
        # Get the data based on the provided list of keys.
        section = copy.deepcopy(self._feature_config)

        for key in key_list:
            section = section[key]

        # If the root data should be included, add the respective keys
        _root_data_white_list = {"multiwfn": ["OMP_STACKSIZE", "NUM_THREADS"]}
        _white_list = _root_data_white_list.get(key_list[0], None)

        # Edge case: ["multiwfn"] as key_list -> only return the root data
        if key_list == ["multiwfn"]:
            if _white_list is not None:
                section = {key: value for key, value in section.items() if key in _white_list}
            else:
                section = {}
            return section

        # Add the root data to section
        if include_root_data is True and _white_list is not None:
            for root_key in _white_list:
                section[root_key] = self._feature_config[key_list[0]][root_key]

        return section
