"""Atom connectivity features from ``Multiwfn``."""

import os
from typing import List, Union

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver
from bonafide.utils.io_ import write_xyz_file_from_coordinates_array


class _Multiwfn3DAtomConnectivity(BaseFeaturizer):
    """Parent feature factory for the 3D atom connectivity Multiwfn features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    connectivity_index_threshold: float

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the Multiwfn connectivity features."""
        # For the connectivity index, no electronic structure data is required. Therefore, it is
        # calculated from the XYZ file.
        # Generate the XYZ file and write it to the electronic_struc_n attribute for input
        # to Multiwfn
        self._write_input_file()
        self._run_multiwfn()
        self._read_output_file()

    def _write_input_file(self) -> None:
        """Write an XYZ file as input for Multiwfn.

        Returns
        -------
        None
        """
        assert self.coordinates is not None  # for type checker
        write_xyz_file_from_coordinates_array(
            elements=self.elements,
            coordinates=self.coordinates,
            file_path=self.conformer_name + ".xyz",
        )

    def _run_multiwfn(self) -> None:
        """Run Multiwfn.

        Returns
        -------
        None
        """
        # Select other functions
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [100]

        # Select connectivity index analysis
        multiwfn_commands.append(9)

        # Set threshold option
        multiwfn_commands.append(self.connectivity_index_threshold)

        # Exit program
        multiwfn_commands.extend(["n", 0, "q"])

        # Set up environment variables
        environment_variables = {
            var: getattr(self, var, None) for var in PROGRAM_ENVIRONMENT_VARIABLES["multiwfn"]
        }

        # Run Multiwfn
        multiwfn_driver(
            cmds=multiwfn_commands,
            input_file_path=self.conformer_name + ".xyz",
            output_file_name=f"{self.__class__.__name__}__{self.conformer_name}",
            environment_variables=environment_variables,
            namespace=self.conformer_name[::-1].split("__", 1)[-1][::-1],
        )

    def _read_output_file(self) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        Returns
        -------
        None
        """
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

        # Find the values and write them to the results dictionary
        for line in multiwfn_output:
            if "Sum of connectivity:" in line and "Sum of integer connectivity:" in line:
                atom_idx = int(line.split()[0])
                splitted = line.split(":")[1:]
                con = float(splitted[0].split("Sum")[0])
                int_con = int(splitted[-1])

                self.results[atom_idx - 1] = {
                    "multiwfn3D-atom-connectivity_index": con,
                    "multiwfn3D-atom-connectivity_index_integer": int_con,
                }


class Multiwfn3DAtomConnectivityIndex(_Multiwfn3DAtomConnectivity):
    """Feature factory for the 3D atom feature "connectivity_index", calculated with multiwfn.

    The index of this feature is 230 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.bond_analysis" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomConnectivity


class Multiwfn3DAtomConnectivityIndexInteger(_Multiwfn3DAtomConnectivity):
    """Feature factory for the 3D atom feature "connectivity_index_integer", calculated with
    multiwfn.

    The index of this feature is 231 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.bond_analysis" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomConnectivity
