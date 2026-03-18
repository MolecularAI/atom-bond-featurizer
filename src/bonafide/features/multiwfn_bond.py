"""Bond order features from ``Multiwfn``."""

import logging
import os
from typing import List, Union

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver
from bonafide.utils.helper_functions import clean_up, get_function_or_method_name
from bonafide.utils.io_ import write_sd_file


class _Multiwfn3DBondOrder(BaseFeaturizer):
    """Parent feature factory for the 3D bond order Multiwfn features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def _run_multiwfn(self, command_list: List[Union[int, str, float]]) -> None:
        """Run Multiwfn.

        Parameters
        ----------
        command_list : List[Union[int, str, float]]
            List of commands to be passed to Multiwfn to select the respective bond order analysis
            method and potential settings.

        Returns
        -------
        None
        """
        # Select bond order analysis
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [9]

        # Add commands from the child class to the Multiwfn command
        multiwfn_commands.extend(command_list)

        # Exit program
        multiwfn_commands.extend([0, "q"])

        # Set up environment variables
        environment_variables = {
            var: getattr(self, var, None) for var in PROGRAM_ENVIRONMENT_VARIABLES["multiwfn"]
        }

        # Run Multiwfn
        multiwfn_driver(
            cmds=multiwfn_commands,
            input_file_path=str(self.electronic_struc_n),
            output_file_name=f"{self.__class__.__name__}__{self.conformer_name}",
            environment_variables=environment_variables,
            namespace=self.conformer_name[::-1].split("__", 1)[-1][::-1],
        )

    def _read_output_file(self, start_string: str, skip_n_lines: int = 1) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        Parameters
        ----------
        start_string : str
            The string that marks the beginning of the relevant section in the output file.
        skip_n_lines : int, optional
            How many lines to skip after the line starting with ``start_string`` before reading the
            actual data, by default 1.

        Returns
        -------
        None
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Check if the output file exists
        _opath = f"{self.__class__.__name__}__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Find relevant positions in the file
        start_idx = None
        for line_idx, line in enumerate(multiwfn_output):
            if line.startswith(start_string):
                start_idx = line_idx + skip_n_lines

        # Check if start_idx_2 was found (this must always be found, start_idx_1 is optional.
        # If start_idx_2 is found, start_idx_1 is also there)
        if start_idx is None:
            self._err = (
                f"output file generated through '{self.__class__.__name__}' does not "
                "contain the requested data; probably the calculation failed. "
                "Check the output file"
            )
            return

        # Save values to results dictionary
        for line in multiwfn_output[start_idx:]:
            if line.strip() != "":
                atom_idx_1 = int(line.split("(")[0].split(":")[-1]) - 1
                atom_idx_2 = int(line.split("(")[1].split(")")[-1]) - 1
                try:
                    value = float(line.split(")")[-1])
                except ValueError:
                    value = float(line.split(":")[-1])

                bond = self.mol.GetBondBetweenAtoms(atom_idx_1, atom_idx_2)
                if bond is None:
                    _namespace = self.conformer_name[::-1].split("__", 1)[-1][::-1]
                    logging.warning(
                        f"'{_namespace}' | {_loc}()\nA bond feature was calculated between atoms "
                        f"'{atom_idx_1}' and '{atom_idx_2}', but no bond is defined between these "
                        "two atoms. Check your input and output to ensure that this is correct "
                        "behavior."
                    )
                else:
                    self.results[bond.GetIdx()] = {self.feature_name: value}
            else:
                break


class Multiwfn3DBondBondOrderFuzzy(_Multiwfn3DBondOrder):
    """Feature factory for the 3D bond feature "bond_order_fuzzy", calculated with multiwfn.

    The index of this feature is 427 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.bond_analysis" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-bond_order_fuzzy`` feature."""
        self._run_multiwfn(command_list=[7, "n"])
        self._read_output_file(start_string=" The total bond order >=")


class Multiwfn3DBondBondOrderLaplacian(_Multiwfn3DBondOrder):
    """Feature factory for the 3D bond feature "bond_order_laplacian", calculated with multiwfn.

    The index of this feature is 428 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.bond_analysis" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-bond_order_laplacian`` feature."""
        self._run_multiwfn(command_list=[8, "n"])
        self._read_output_file(start_string=" The bond orders >=")


class Multiwfn3DBondBondOrderMayer(_Multiwfn3DBondOrder):
    """Feature factory for the 3D bond feature "bond_order_mayer", calculated with multiwfn.

    The index of this feature is 429 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.bond_analysis" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-bond_order_mayer`` feature."""
        self._run_multiwfn(command_list=[1, "n"])
        self._read_output_file(start_string=" Bond orders with absolute value >=")


class Multiwfn3DBondBondOrderMulliken(_Multiwfn3DBondOrder):
    """Feature factory for the 3D bond feature "bond_order_mulliken", calculated with multiwfn.

    The index of this feature is 430 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.bond_analysis" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-bond_order_mulliken`` feature."""
        self._run_multiwfn(command_list=[4, "n"])
        self._read_output_file(start_string=" Bond orders with absolute value >=")


class Multiwfn3DBondBondOrderWiberg(_Multiwfn3DBondOrder):
    """Feature factory for the 3D bond feature "bond_order_wiberg", calculated with multiwfn.

    The index of this feature is 431 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.bond_analysis" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-bond_order_wiberg`` feature."""
        self._run_multiwfn(command_list=[3, "n"])
        self._read_output_file(start_string=" Bond orders with absolute value >=")


class Multiwfn3DBondIntrinsicBondStrengthIndex(_Multiwfn3DBondOrder):
    """Feature factory for the 3D bond feature "intrinsic_bond_strength_index", calculated with
    multiwfn.

    The index of this feature is 445 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.bond_analysis" in the _feature_config.toml file.
    """

    ibsi_grid: str
    ibis_igm_type: str

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-intrinsic_bond_strength_index`` feature."""
        # For the IBSI, electronic structure data is not necessarily required. It can be calculated
        # with an SD or electronic structure file as input.

        # Select IBSI analysis
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [10]

        # No electronic structure data available
        _no_el_struc = False
        if self.electronic_struc_n is None:
            # Check if electronic structure data is available
            if self.ibis_igm_type == "hirshfeld":
                self._err = (
                    f"for calculating the '{self.feature_name}' feature with the IGM type "
                    "'hirshfeld', electronic structure data is required but is not available. "
                    "Attach precomputed electronic structure data or calculate it from scratch"
                )
                return

            # Write temporary SDF input
            write_sd_file(mol=self.mol, file_path=f"{self.conformer_name}.sdf")
            self.electronic_struc_n = f"{self.conformer_name}.sdf"
            _no_el_struc = True

        # Select type of IGM
        multiwfn_commands.append(2)
        if self.ibis_igm_type == "promolecular":
            multiwfn_commands.append(1)
        if self.ibis_igm_type == "hirshfeld":
            multiwfn_commands.append(2)

        # Start calculation command
        multiwfn_commands.append(1)

        # Select grid quality (calculation runs automatically after selection) and return
        multiwfn_commands.extend([self.ibsi_grid, 0])

        # Run Multiwfn
        self._run_multiwfn(command_list=multiwfn_commands)

        # Reset self.electronic_struc_n (just to be safe)
        if _no_el_struc is True:
            self.electronic_struc_n = None

        # Read output file
        self._read_output_file(
            start_string=' Note: "Dist" is distance between the two atoms in Angstrom, '
            "Int(dg_pair)",
            skip_n_lines=2,
        )

        # Clean up (remove temporary SDF file)
        clean_up(to_be_removed=[f"{self.conformer_name}.sdf"])
