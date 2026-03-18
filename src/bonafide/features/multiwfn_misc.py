"""Energy index atom feature from ``Multiwfn``."""

import os
from typing import List, Union

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver


class Multiwfn3DAtomEnergyIndex(BaseFeaturizer):
    """Feature factory for the 3D atom feature "energy_index", calculated with multiwfn.

    The index of this feature is 232 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-energy_index`` feature."""
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
        multiwfn_commands = [200]

        # Select energy index analysis
        multiwfn_commands.append(12)

        # Select atom
        multiwfn_commands.append(self.atom_bond_idx + 1)  # Multiwfn uses 1-based indexing

        # Exit program
        multiwfn_commands.extend([0, 0, "q"])

        # Set up environment variables
        environment_variables = {
            var: getattr(self, var, None) for var in PROGRAM_ENVIRONMENT_VARIABLES["multiwfn"]
        }

        # Run Multiwfn
        multiwfn_driver(
            cmds=multiwfn_commands,
            input_file_path=str(self.electronic_struc_n),
            output_file_name=f"{self.__class__.__name__}__{self.conformer_name}__"
            f"atom-{self.atom_bond_idx}",
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
        _opath = f"{self.__class__.__name__}__{self.conformer_name}__atom-{self.atom_bond_idx}.out"
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
        for line in multiwfn_output:
            if line.startswith(" The EI index:") and "a.u." in line:
                val = float(line.split(":")[-1].split()[0])
                self.results[self.atom_bond_idx] = {self.feature_name: val}
                break
