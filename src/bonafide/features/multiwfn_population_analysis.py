"""Base class for all population analyses features from ``Multiwfn``."""

import os
import re
from typing import Dict, List, Union

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver
from bonafide.utils.helper_functions import clean_up


class _Multiwfn3DAtomPopulationAnalysis(BaseFeaturizer):
    """Parent feature factory for the 3D atom population analysis Multiwfn features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def _run_multiwfn(
        self,
        command_list: List[Union[int, float, str]],
        from_eem: bool = False,
        from_resp_chelpg: bool = False,
    ) -> None:
        """Run Multiwfn.

        Parameters
        ----------
        command_list : List[Union[int, float, str]]
            The list of commands to be passed to Multiwfn to select the respective population
            analysis method and options.
        from_eem : bool, optional
            Whether the Multiwfn run is for EEM charges. Default is ``False``.
        from_resp_chelpg : bool, optional
            Whether the Multiwfn run is for RESP charges. Default is ``False``.

        Returns
        -------
        None
        """
        # Select population analysis
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [7]

        # Add commands from the child class to the Multiwfn command
        multiwfn_commands.extend(command_list)

        # Exit program
        multiwfn_commands.append("n")
        if from_eem is True:
            multiwfn_commands.append(-1)
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
            modify_ispecial=from_resp_chelpg,
        )

    def _read_output_file(self, feature_name: str) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used to read the data for the Becke, CM5, scaled CM5, Hirsheld, corrected
        Hirshfeld, and Voronoi deformation density charges.

        Parameters
        ----------
        feature_name : str
            The name of the feature.

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

        # Find relevant position in the file
        start_idx = None
        for line_idx, line in enumerate(multiwfn_output):
            if line.strip().startswith("Final atomic charges"):
                start_idx = line_idx + 1
                break

        # Check if start_idx was found
        if start_idx is None:
            self._err = (
                f"output file generated through '{self.__class__.__name__}' does not "
                "contain the requested data; probably the calculation failed. "
                "Check the output file"
            )
            return

        # Save values to results dictionary
        for line in multiwfn_output[start_idx:]:
            if line.strip() == "":
                break

            val = float(line.split("):")[-1])
            atom_idx = int(line.split("Atom")[-1].strip().split("(")[0])
            self.results[atom_idx - 1] = {feature_name: val}

    def _read_output_file2(self, scheme_name: str) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is only used for closed-shell molecules (multiplicity = 1). The respective
        method for open-shell molecules is ``_read_output_file3()``. ``_read_output_file2()``
        is used to read the data for the Lowdin, Mulliken and its respective modified versions
        partial charges as well as for the data for the individual Lowdin and Mulliken orbital
        population features.

        Parameters
        ----------
        scheme_name : str
            The name of the population analysis method.

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

        # Find relevant positions in the file
        start_idx_1 = None
        start_idx_2 = None
        for line_idx, line in enumerate(multiwfn_output):
            if line == " Population of each type of angular moment orbitals:\n":
                start_idx_1 = line_idx + 1
            if line == " Population of atoms:\n":
                start_idx_2 = line_idx + 1

        if start_idx_2 is None:
            for line_idx, line in enumerate(multiwfn_output):
                if line == " 20 Minimal Basis Iterative Stockholder (MBIS) charge\n":
                    start_idx_2 = line_idx + 1
                    break

        # Check if start_idx_2 was found (this must always be found, start_idx_1 is optional.
        # If start_idx_2 is found, start_idx_1 is also there in case required)
        if start_idx_2 is None:
            self._err = (
                f"output file generated through '{self.__class__.__name__}' does not "
                "contain the requested data; probably the calculation failed. Check the "
                "output file"
            )
            return

        # Save values to results dictionary
        re_s = re.compile(r"s:\s*([-\d.]+)\s+")
        re_p = re.compile(r"p:\s*([-\d.]+)\s+")
        re_d = re.compile(r"d:\s*([-\d.]+)\s+")
        re_f = re.compile(r"f:\s*([-\d.]+)\s+")
        re_g = re.compile(r"g:\s*([-\d.]+)\s+")
        re_h = re.compile(r"h:\s*([-\d.]+)\s+")

        re_pop = re.compile(r"Population:\s*([-\d.]+)\s+")
        re_charge = re.compile(r"charge:\s*([-\d.]+)\s+")

        if start_idx_1 is not None:
            for line in multiwfn_output[start_idx_1:]:
                if line.strip().startswith("Sum"):
                    break

                atom_idx = int(line.split("Atom ")[-1].strip().split("(")[0])
                s_pop = float(re_s.findall(line)[0])
                p_pop = float(re_p.findall(line)[0])
                d_pop = float(re_d.findall(line)[0])
                f_pop = float(re_f.findall(line)[0])
                g_pop = float(re_g.findall(line)[0])
                h_pop = float(re_h.findall(line)[0])
                self.results[atom_idx - 1] = {
                    f"multiwfn3D-atom-population_{scheme_name}_s": s_pop,
                    f"multiwfn3D-atom-population_{scheme_name}_p": p_pop,
                    f"multiwfn3D-atom-population_{scheme_name}_d": d_pop,
                    f"multiwfn3D-atom-population_{scheme_name}_f": f_pop,
                    f"multiwfn3D-atom-population_{scheme_name}_g": g_pop,
                    f"multiwfn3D-atom-population_{scheme_name}_h": h_pop,
                }

        for line in multiwfn_output[start_idx_2:]:
            if not line.startswith(" Total net charge:"):
                atom_idx = int(line.split("Atom ")[-1].strip().split("(")[0])
                pop = float(re_pop.findall(line)[0])
                charge = float(re_charge.findall(line)[0])
                if atom_idx - 1 not in self.results:
                    self.results[atom_idx - 1] = {}
                self.results[atom_idx - 1][f"multiwfn3D-atom-population_{scheme_name}"] = pop
                self.results[atom_idx - 1][f"multiwfn3D-atom-partial_charge_{scheme_name}"] = charge
            else:
                break

    def _read_output_file3(self, scheme_name: str) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is only used for open-shell molecules (multiplicity != 1). The respective
        method for closed-shell molecules is ``_read_output_file2()``. ``_read_output_file3()``
        is used to read the data for the Lowdin, Mulliken and its respective modified versions
        partial charges as well as for the data for the individual Lowdin and Mulliken orbital
        population features. Additionally, it is used to read the spin population data.

        Parameters
        ----------
        scheme_name : str
            The name of the population analysis method.

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

        # Find relevant positions in the file
        start_idx_1 = None
        start_idx_2 = None
        for line_idx, line in enumerate(multiwfn_output):
            if line == " Population of each type of angular moment atomic orbitals:\n":
                start_idx_1 = line_idx + 2
            if line == " Population of atoms:\n":
                start_idx_2 = line_idx + 2

        if start_idx_2 is None:
            for line_idx, line in enumerate(multiwfn_output):
                if line == " 20 Minimal Basis Iterative Stockholder (MBIS) charge\n":
                    start_idx_2 = line_idx + 2
                    break

        # Check if start_idx_2 was found (this must always be found, start_idx_1 is optional.
        # If start_idx_2 is found, start_idx_1 is also there)
        if start_idx_2 is None:
            self._err = (
                f"output file generated through '{self.__class__.__name__}' does not "
                "contain the requested data; probably the calculation failed. Check the "
                "output file"
            )
            return

        # Extract the data for the individual orbital populations
        if start_idx_1 is not None:
            helper_dict: Dict[int, Dict[str, Dict[str, float]]] = {}
            for line in multiwfn_output[start_idx_1:]:
                if line == " \n":
                    break

                if "(" in line:
                    atom_idx = int(line.split("(")[0])
                    ang_moment = line.split(")")[-1].strip().split()[0]
                else:
                    ang_moment = line.split()[0]

                _splitted = line.split()
                spin_pop = float(_splitted[-1])
                pop = float(_splitted[-2])
                pop_beta = float(_splitted[-3])
                pop_alpha = float(_splitted[-4])

                if atom_idx - 1 not in helper_dict:
                    helper_dict[atom_idx - 1] = {}
                helper_dict[atom_idx - 1][ang_moment] = {
                    f"multiwfn3D-atom-population_{scheme_name}_{ang_moment}": pop,
                    f"multiwfn3D-atom-population_{scheme_name}_{ang_moment}_alpha": pop_alpha,
                    f"multiwfn3D-atom-population_{scheme_name}_{ang_moment}_beta": pop_beta,
                    f"multiwfn3D-atom-spin_population_{scheme_name}_{ang_moment}": spin_pop,
                }

            # Write the data to the results dictionary
            for atom_idx, data in helper_dict.items():
                if atom_idx not in self.results:
                    self.results[atom_idx] = {}

                for ang_moment in ["s", "p", "d", "f", "g", "h"]:
                    if ang_moment not in data:
                        self.results[atom_idx][
                            f"multiwfn3D-atom-population_{scheme_name}_{ang_moment}"
                        ] = 0.0
                        self.results[atom_idx][
                            f"multiwfn3D-atom-population_{scheme_name}_{ang_moment}_alpha"
                        ] = 0.0
                        self.results[atom_idx][
                            f"multiwfn3D-atom-population_{scheme_name}_{ang_moment}_beta"
                        ] = 0.0
                        self.results[atom_idx][
                            f"multiwfn3D-atom-spin_population_{scheme_name}_{ang_moment}"
                        ] = 0.0
                        continue

                    for feature_name, value in data[ang_moment].items():
                        self.results[atom_idx][feature_name] = value

        # Extract the total atom populations and charges and write them to the results dictionary
        for line in multiwfn_output[start_idx_2:]:
            if line.strip().startswith("Total net charge:"):
                break

            atom_idx = int(line.split("(")[0])

            _splitted = line.split()
            charge = float(_splitted[-1])
            spin = float(_splitted[-2])
            pop_beta = float(_splitted[-3])
            pop_alpha = float(_splitted[-4])

            if atom_idx - 1 not in self.results:
                self.results[atom_idx - 1] = {}

            self.results[atom_idx - 1][f"multiwfn3D-atom-partial_charge_{scheme_name}"] = charge
            self.results[atom_idx - 1][f"multiwfn3D-atom-population_{scheme_name}"] = round(
                number=pop_alpha + pop_beta, ndigits=5
            )
            self.results[atom_idx - 1][f"multiwfn3D-atom-population_{scheme_name}_alpha"] = (
                pop_alpha
            )
            self.results[atom_idx - 1][f"multiwfn3D-atom-population_{scheme_name}_beta"] = pop_beta
            self.results[atom_idx - 1][f"multiwfn3D-atom-spin_population_{scheme_name}"] = spin

    def _read_output_file4(self, feature_name: str) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used to read the data for the CHELPG, Merz-Kollmann, and RESP charges.

        Parameters
        ----------
        feature_name : str
            The name of the feature.

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

        # Find relevant position in the file
        start_idx = None
        for line_idx, line in enumerate(multiwfn_output):
            if line == "   Center       Charge\n":
                start_idx = line_idx + 1

        # Check if start_idx was found
        if start_idx is None:
            self._err = (
                f"output file generated through '{self.__class__.__name__}' does not "
                "contain the requested data; probably the calculation failed. Check the "
                "output file"
            )
            return

        # Save values to results dictionary
        for line in multiwfn_output[start_idx:]:
            if line.startswith(" Sum of charges:") is True:
                break

            val = float(line.split(")")[-1])
            atom_idx = int(line.split("(")[0])
            self.results[atom_idx - 1] = {feature_name: val}

    def _read_output_file5(self, feature_name: str) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used to read the data for the EEM charges.

        Parameters
        ----------
        feature_name : str
            The name of the feature.

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

        # Parse the output file
        for line in multiwfn_output:
            # Error handling if EEM parameters are unavailable for a specific atom type
            if "Error: Parameter for atom" in line:
                _splitted = line.split("(")
                atom_idx = int(_splitted[0].split()[-1]) - 1
                atom_symbol = _splitted[-1].split(")")[0].strip()

                self._err = (
                    f"EEM parameters for atom with index {atom_idx} ({atom_symbol}) are "
                    "unavailable. Choose a different set of EEM parameters in "
                    "'multiwfn.population.eem_parameters' (see the print_options() and "
                    "set_options() methods)"
                )
                clean_up(to_be_removed=["*.sdf"])
                return

            # Get the values and write them to the results dictionary
            if line.strip().startswith("Electronegativity:"):
                break

            if line.strip().startswith("EEM charge of atom"):
                atom_idx = int(line.split("(")[0].split("atom")[-1])
                val = float(line.split(":")[-1])
                self.results[atom_idx - 1] = {feature_name: val}

        # Clean up
        clean_up(to_be_removed=["*.sdf"])
