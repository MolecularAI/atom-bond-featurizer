"""Miscellaneous bond features from ``Multiwfn``."""

import os
from typing import List, Union

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver
from bonafide.utils.io_ import write_xyz_file_from_coordinates_array


class _Multiwfn3DBond(BaseFeaturizer):
    """Parent feature factory for the 3D bond Multiwfn features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the Multiwfn bond features."""
        # For the data calculated here, no electronic structure data is required. Therefore, it is
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

        # Select structure info data
        multiwfn_commands.append(21)

        # Define bond
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        multiwfn_commands.append(f"{begin_atom_idx + 1},{end_atom_idx + 1}")

        # Exit program
        multiwfn_commands.extend(["q", 0, "q"])

        # Set up environment variables
        environment_variables = {
            var: getattr(self, var, None) for var in PROGRAM_ENVIRONMENT_VARIABLES["multiwfn"]
        }

        # Run Multiwfn
        multiwfn_driver(
            cmds=multiwfn_commands,
            input_file_path=self.conformer_name + ".xyz",
            output_file_name=f"Multiwfn3DBondMiscInfo__{self.conformer_name}__"
            f"bond-{self.atom_bond_idx}",
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
        _opath = f"Multiwfn3DBondMiscInfo__{self.conformer_name}__bond-{self.atom_bond_idx}.out"
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
        for line_idx, line in enumerate(multiwfn_output):
            if line.startswith(" Center of mass (X/Y/Z):"):
                splitted = line.split()
                x = splitted[-4].strip()
                y = splitted[-3].strip()
                z = splitted[-2].strip()
                self.results[self.atom_bond_idx] = {
                    "multiwfn3D-bond-coordinates_center_of_mass": ",".join([x, y, z]),
                }

            if line.startswith(" Center of nuclear charges (X/Y/Z):"):
                splitted = line.split()
                x = splitted[-4].strip()
                y = splitted[-3].strip()
                z = splitted[-2].strip()
                self.results[self.atom_bond_idx][
                    "multiwfn3D-bond-coordinates_center_of_nuclear_charges"
                ] = ",".join([x, y, z])

            if line.startswith(" Electrostatic interaction energy between nuclear charges:"):
                splitted = multiwfn_output[line_idx + 1].split()
                self.results[self.atom_bond_idx][
                    "multiwfn3D-bond-electrostatic_interaction_energy_nuclear_charges"
                ] = float(splitted[-2])

            if line.startswith(" Radius of gyration:"):
                val = float(line.split(":")[-1].split()[0])
                self.results[self.atom_bond_idx]["multiwfn3D-bond-gyration_radius"] = val


class Multiwfn3DBondCoordinatesCenterOfMass(_Multiwfn3DBond):
    """Feature factory for the 3D bond feature "coordinates_center_of_mass", calculated with
    multiwfn.

    The index of this feature is 432 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBond


class Multiwfn3DBondCoordinatesCenterOfNuclearCharges(_Multiwfn3DBond):
    """Feature factory for the 3D bond feature "coordinates_center_of_nuclear_charges",
    calculated with multiwfn.

    The index of this feature is 433 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBond


class Multiwfn3DBondElectrostaticInteractionEnergyNuclearCharges(_Multiwfn3DBond):
    """Feature factory for the 3D bond feature
    "electrostatic_interaction_energy_nuclear_charges", calculated with multiwfn.

    The index of this feature is 434 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBond


class Multiwfn3DBondGyrationRadius(_Multiwfn3DBond):
    """Feature factory for the 3D bond feature "gyration_radius", calculated with multiwfn.

    The index of this feature is 439 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBond
