"""Hellmann-Feynman force features from ``Multiwfn``."""

import os
from typing import List, Union

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver


class _Multiwfn3DAtomForce(BaseFeaturizer):
    """Parent feature factory for the 3D atom Multiwfn Hellmann-Feynman force features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the Multiwfn Hellmann-Feynman force features."""
        self._run_multiwfn()
        self._read_output_file()

    def _run_multiwfn(self) -> None:
        """Run Multiwfn.

        Returns
        -------
        None
        """
        # Select other functions
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [100]

        # Select Hellmann-Feynman forces
        multiwfn_commands.append(20)

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
            output_file_name=f"Multiwfn3DAtomForceHellmannFeynman__{self.conformer_name}",
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
        _opath = f"Multiwfn3DAtomForceHellmannFeynman__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Find the value and write it to the results dictionary
        for line_idx, line in enumerate(multiwfn_output):
            # Electron contribution
            if all(
                [
                    "Hellmann-Feynman" in line,
                    "forces" in line,
                    "contributed " in line,
                    "by" in line,
                    "electrons" in line,
                ]
            ):
                for line_a in multiwfn_output[line_idx + 2 :]:
                    if line_a.strip() == "":
                        break
                    atom_idx = int(line_a.split("(")[0])
                    val = float(line_a.split()[-1])
                    self.results[atom_idx - 1] = {
                        "multiwfn3D-atom-force_hellmann_feynman_electrons": val
                    }

            # Nuclear contribution
            if all(
                [
                    "Hellmann-Feynman" in line,
                    "forces" in line,
                    "contributed " in line,
                    "by" in line,
                    "nuclear" in line,
                    "charges" in line,
                ]
            ):
                for line_a in multiwfn_output[line_idx + 2 :]:
                    if line_a.strip() == "":
                        break
                    atom_idx = int(line_a.split("(")[0])
                    val = float(line_a.split()[-1])
                    self.results[atom_idx - 1][
                        "multiwfn3D-atom-force_hellmann_feynman_nuclear_charges"
                    ] = val

            # Total
            if all(
                [
                    "Total " in line,
                    "Hellmann-Feynman" in line,
                    "forces" in line,
                ]
            ):
                for line_a in multiwfn_output[line_idx + 2 :]:
                    if line_a.strip() == "":
                        break
                    atom_idx = int(line_a.split("(")[0])
                    val = float(line_a.split()[-1])
                    self.results[atom_idx - 1]["multiwfn3D-atom-force_hellmann_feynman"] = val


class Multiwfn3DAtomForceHellmannFeynman(_Multiwfn3DAtomForce):
    """Feature factory for the 3D atom feature "force_hellmann_feynman", calculated with
    multiwfn.

    The index of this feature is 233 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomForce


class Multiwfn3DAtomForceHellmannFeynmanElectrons(_Multiwfn3DAtomForce):
    """Feature factory for the 3D atom feature "force_hellmann_feynman_electrons", calculated
    with multiwfn.

    The index of this feature is 234 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomForce


class Multiwfn3DAtomForceHellmannFeynmanNuclearCharges(_Multiwfn3DAtomForce):
    """Feature factory for the 3D atom feature "force_hellmann_feynman_nuclear_charges",
    calculated with multiwfn.

    The index of this feature is 235 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomForce
