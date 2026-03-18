"""Fukui indices calculated with ``Multiwfn``."""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

from bonafide.features.multiwfn_partial_charge import (
    Multiwfn3DAtomPartialChargeBecke,
    Multiwfn3DAtomPartialChargeChelpg,
    Multiwfn3DAtomPartialChargeCm5,
    Multiwfn3DAtomPartialChargeCm5Scaled1point2,
    Multiwfn3DAtomPartialChargeCorrectedHirshfeld,
    Multiwfn3DAtomPartialChargeHirshfeld,
    Multiwfn3DAtomPartialChargeLowdin,
    Multiwfn3DAtomPartialChargeMerzKollmann,
    Multiwfn3DAtomPartialChargeMulliken,
    Multiwfn3DAtomPartialChargeMullikenBickelhaupt,
    Multiwfn3DAtomPartialChargeMullikenRosSchuit,
    Multiwfn3DAtomPartialChargeMullikenStoutPolitzer,
    Multiwfn3DAtomPartialChargeRespChelpgOneStage,
    Multiwfn3DAtomPartialChargeRespChelpgTwoStage,
    Multiwfn3DAtomPartialChargeRespMerzKollmannOneStage,
    Multiwfn3DAtomPartialChargeRespMerzKollmannTwoStage,
    Multiwfn3DAtomPartialChargeVdd,
)
from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver


class _Multiwfn3DAtomCdftCondensedOrbitalWeightedFukui(BaseFeaturizer):
    """Parent feature factory for the 3D atom Multiwfn condensed orbital-weighted Fukui index
    features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    ow_delta: float

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def _run_multiwfn(self) -> None:
        """Run Multiwfn.

        Returns
        -------
        None
        """
        # Select C-DFT analysis
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [22]

        # Set delta parameter for orbital weighting
        multiwfn_commands.extend([4, self.ow_delta])

        # Run analysis
        multiwfn_commands.append(6)

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

        # Find relevant position in the file
        start_idx = None
        for line_idx, line in enumerate(multiwfn_output):
            if all(
                [
                    "Atom index" in line,
                    "OW f+" in line,
                    "OW f-" in line,
                    "OW f0" in line,
                    "OW DD" in line,
                ]
            ):
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
            if line.strip() == "":
                break
            atom_idx = int(line.split("(")[0])

            splitted = line.split()
            ow_f_plus = float(splitted[-4])
            ow_f_minus = float(splitted[-3])
            ow_f_zero = float(splitted[-2])
            ow_f_dual = float(splitted[-1])

            self.results[atom_idx - 1] = {
                "multiwfn3D-atom-cdft_condensed_orbital_weighted_fukui_plus": ow_f_plus,
                "multiwfn3D-atom-cdft_condensed_orbital_weighted_fukui_minus": ow_f_minus,
                "multiwfn3D-atom-cdft_condensed_orbital_weighted_fukui_zero": ow_f_zero,
                "multiwfn3D-atom-cdft_condensed_orbital_weighted_fukui_dual": ow_f_dual,
            }


class Multiwfn3DAtomCdftCondensedOrbitalWeightedFukuiDual(
    _Multiwfn3DAtomCdftCondensedOrbitalWeightedFukui
):
    """Feature factory for the 3D atom feature "cdft_condensed_orbital_weighted_fukui_dual",
    calculated with multiwfn.

    The index of this feature is 204 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_condensed_orbital_weighted_fukui_dual``
        feature."""
        # Feature is not defined for open-shell molecules
        if self.multiplicity != 1:
            self.results[0] = {self.feature_name: "_inaccessible"}
            return

        self._run_multiwfn()
        self._read_output_file()


class Multiwfn3DAtomCdftCondensedOrbitalWeightedFukuiMinus(
    _Multiwfn3DAtomCdftCondensedOrbitalWeightedFukui
):
    """Feature factory for the 3D atom feature "cdft_condensed_orbital_weighted_fukui_minus",
    calculated with multiwfn.

    The index of this feature is 205 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_condensed_orbital_weighted_fukui_minus``
        feature."""
        # Feature is not defined for open-shell molecules
        if self.multiplicity != 1:
            self.results[0] = {self.feature_name: "_inaccessible"}
            return

        self._run_multiwfn()
        self._read_output_file()


class Multiwfn3DAtomCdftCondensedOrbitalWeightedFukuiPlus(
    _Multiwfn3DAtomCdftCondensedOrbitalWeightedFukui
):
    """Feature factory for the 3D atom feature "cdft_condensed_orbital_weighted_fukui_plus",
    calculated with multiwfn.

    The index of this feature is 206 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_condensed_orbital_weighted_fukui_plus``
        feature."""
        # Feature is not defined for open-shell molecules
        if self.multiplicity != 1:
            self.results[0] = {self.feature_name: "_inaccessible"}
            return

        self._run_multiwfn()
        self._read_output_file()


class Multiwfn3DAtomCdftCondensedOrbitalWeightedFukuiZero(
    _Multiwfn3DAtomCdftCondensedOrbitalWeightedFukui
):
    """Feature factory for the 3D atom feature "cdft_condensed_orbital_weighted_fukui_zero",
    calculated with multiwfn.

    The index of this feature is 207 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_condensed_orbital_weighted_fukui_zero``
        feature."""
        # Feature is not defined for open-shell molecules
        if self.multiplicity != 1:
            self.results[0] = {self.feature_name: "_inaccessible"}
            return

        self._run_multiwfn()
        self._read_output_file()


class _Multiwfn3DAtomCdftCondensedFukui(BaseFeaturizer):
    """Parent feature factory for the 3D atom Multiwfn Fukui index features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    ensemble_dimensionality: str
    feature_cache_n_minus1: List[Dict[str, Dict[int, Optional[Union[str, bool, int, float]]]]]
    feature_cache_n_plus1: List[Dict[str, Dict[int, Optional[Union[str, bool, int, float]]]]]
    feature_dimensionality: int
    iterable_option: str
    multiplicity: int
    NUM_THREADS: Optional[int]
    OMP_STACKSIZE: Optional[str]

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _check_electronic_structure_data(
        self, el_struc_data: Optional[str], data_name: str
    ) -> None:
        """Check if the required electronic structure data is available.

        Parameters
        ----------
        el_struc_data : Optional[str]
            The electronic structure data to check. It is either ``None`` (not available) or the
            path to the electronic structure data file.
        data_name : str
            The identification string of the electronic structure data to check (used for logging).

        Returns
        -------
        None
        """
        if el_struc_data is None:
            self._err = (
                f"for requesting data from '{self.__class__.__name__}', electronic structure "
                f"data for the {data_name} is required but is not available. Attach precomputed "
                "electronic structure data or calculate it from scratch"
            )

    def _run_calculation(
        self, **kwargs: Any
    ) -> Tuple[Optional[Union[int, float, bool, str]], Optional[str]]:
        """Run Multiwfn to calculate the atomic partial charges required to calculate the Fukui
        indices.

        This is done by initializing a "child" feature calculation pipeline as for all other
        features. After this is completed, the Fukui indices are calculated.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to the ``calculate()`` method of the feature
            factory for calculating the partial charges. These parameters are set as attributes
            of the "child" feature factory instance.

        Returns
        -------
        Tuple[Optional[Union[int, float, bool, str]], Optional[str]]
            The calculated partial charge and the error message from the "child" feature factory
            call. The error message is ``None`` if the "child" feature factory did not encounter
            an error.
        """
        # Check if the "child" output files folder already exists
        if os.path.isdir("_temp_out_files_child_dir"):
            shutil.rmtree("_temp_out_files_child_dir")
        os.mkdir("_temp_out_files_child_dir")

        # Mapping dictionary to use the correct feature factory of the selected charge scheme.
        charge_scheme_mapping = {
            "becke": (Multiwfn3DAtomPartialChargeBecke, "multiwfn3D-atom-partial_charge_becke"),
            "chelpg": (Multiwfn3DAtomPartialChargeChelpg, "multiwfn3D-atom-partial_charge_chelpg"),
            "cm5": (Multiwfn3DAtomPartialChargeCm5, "multiwfn3D-atom-partial_charge_cm5"),
            "scaled_cm5": (
                Multiwfn3DAtomPartialChargeCm5Scaled1point2,
                "multiwfn3D-atom-partial_charge_cm5_scaled_1point2",
            ),
            "corrected_hirshfeld": (
                Multiwfn3DAtomPartialChargeCorrectedHirshfeld,
                "multiwfn3D-atom-partial_charge_corrected_hirshfeld",
            ),
            "hirshfeld": (
                Multiwfn3DAtomPartialChargeHirshfeld,
                "multiwfn3D-atom-partial_charge_hirshfeld",
            ),
            "lowdin": (Multiwfn3DAtomPartialChargeLowdin, "multiwfn3D-atom-partial_charge_lowdin"),
            "merz_kollmann": (
                Multiwfn3DAtomPartialChargeMerzKollmann,
                "multiwfn3D-atom-partial_charge_merz_kollmann",
            ),
            "mulliken": (
                Multiwfn3DAtomPartialChargeMulliken,
                "multiwfn3D-atom-partial_charge_mulliken",
            ),
            "mulliken_bickelhaupt": (
                Multiwfn3DAtomPartialChargeMullikenBickelhaupt,
                "multiwfn3D-atom-partial_charge_mulliken_bickelhaupt",
            ),
            "mulliken_ros_schuit": (
                Multiwfn3DAtomPartialChargeMullikenRosSchuit,
                "multiwfn3D-atom-partial_charge_mulliken_ros_schuit",
            ),
            "mulliken_stout_politzer": (
                Multiwfn3DAtomPartialChargeMullikenStoutPolitzer,
                "multiwfn3D-atom-partial_charge_mulliken_stout_politzer",
            ),
            "resp_chelpg_one_stage": (
                Multiwfn3DAtomPartialChargeRespChelpgOneStage,
                "multiwfn3D-atom-partial_charge_resp_chelpg_one_stage",
            ),
            "resp_chelpg_two_stage": (
                Multiwfn3DAtomPartialChargeRespChelpgTwoStage,
                "multiwfn3D-atom-partial_charge_resp_chelpg_two_stage",
            ),
            "resp_merz_kollmann_one_stage": (
                Multiwfn3DAtomPartialChargeRespMerzKollmannOneStage,
                "multiwfn3D-atom-partial_charge_resp_merz_kollmann_one_stage",
            ),
            "resp_merz_kollmann_two_stage": (
                Multiwfn3DAtomPartialChargeRespMerzKollmannTwoStage,
                "multiwfn3D-atom-partial_charge_resp_kollmann_two_stage",
            ),
            "vdd": (Multiwfn3DAtomPartialChargeVdd, "multiwfn3D-atom-partial_charge_vdd"),
        }

        # Calculate the atomic charges required to get the Fukui indices
        feature_factory, feature_name = charge_scheme_mapping[kwargs["charge_scheme"]]
        new_params = {name: value for name, value in kwargs.items()}
        new_params["feature_name"] = feature_name

        calc_feature = feature_factory()
        feature_value, error_message = calc_feature(**new_params)

        # Handle the output files
        if kwargs["_keep_output_files"] is True:
            self._save_output_files2()
        shutil.rmtree("_temp_out_files_child_dir")

        # Return the feature value and the error message of the "child" feature calculation
        # process, which is the calculation of the partial atomic charges
        return feature_value, error_message

    def _save_output_files2(self) -> None:
        """Save the generated output files of the "child" feature factory.

        Returns
        -------
        None
        """
        for item in os.listdir("_temp_out_files_child_dir"):
            source = os.path.join("_temp_out_files_child_dir", item)
            dest = os.path.join(os.path.dirname(os.getcwd()), item)
            if os.path.isfile(source):
                shutil.copy2(source, dest)


class Multiwfn3DAtomCdftCondensedFukuiDual(_Multiwfn3DAtomCdftCondensedFukui):
    """Feature factory for the 3D atom feature "cdft_condensed_fukui_dual", calculated with
    multiwfn.

    The index of this feature is 200 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_condensed_fukui_dual`` feature."""
        # Check if both (radical anion and cation) electronic structure data is available
        self._check_electronic_structure_data(
            el_struc_data=self.electronic_struc_n_plus1,
            data_name="radical anion (actual molecule plus one electron)",
        )
        if self._err is not None:
            return

        self._check_electronic_structure_data(
            el_struc_data=self.electronic_struc_n_minus1,
            data_name="radical cation (actual molecule minus one electron)",
        )
        if self._err is not None:
            return

        # Modify feature_name to also include the name of the charge scheme
        self.feature_name = f"{self.feature_name}__{self.iterable_option}"

        # Get charges of the actual molecule
        feature_value_n, error_message_n = self._run_calculation(
            OMP_STACKSIZE=self.OMP_STACKSIZE,
            NUM_THREADS=self.NUM_THREADS,
            charge_scheme=self.iterable_option,
            feature_cache=self.feature_cache,
            feature_dimensionality=self.feature_dimensionality,
            ensemble_dimensionality=self.ensemble_dimensionality,
            conformer_idx=self.conformer_idx,
            conformer_name=self.conformer_name,
            electronic_struc_n=self.electronic_struc_n,
            feature_type=self.feature_type,
            mol=self.mol,
            multiplicity=self.multiplicity,
            atom_bond_idx=self.atom_bond_idx,
            _keep_output_files=self._keep_output_files,
        )

        if error_message_n is not None:
            self._err = error_message_n
            return

        # Calculate multiplicity of radical anion
        if self.multiplicity == 1:
            multiplicity = 2
        else:
            multiplicity = self.multiplicity - 1

        # Get charges of the radical anion (n+1 state)
        feature_value_n_plus1, error_message_n_plus1 = self._run_calculation(
            OMP_STACKSIZE=self.OMP_STACKSIZE,
            NUM_THREADS=self.NUM_THREADS,
            charge_scheme=self.iterable_option,
            feature_cache=self.feature_cache_n_plus1,
            feature_dimensionality=self.feature_dimensionality,
            ensemble_dimensionality=self.ensemble_dimensionality,
            conformer_idx=self.conformer_idx,
            conformer_name=f"{self.conformer_name}__n+1",
            electronic_struc_n=self.electronic_struc_n_plus1,  # pass electronic structure data for n+1 state to Multiwfn
            feature_type=self.feature_type,
            mol=self.mol,
            multiplicity=multiplicity,
            atom_bond_idx=self.atom_bond_idx,
            _keep_output_files=self._keep_output_files,
        )

        if error_message_n_plus1 is not None:
            self._err = error_message_n_plus1
            return

        # Calculate multiplicity of radical cation
        if self.multiplicity == 1:
            multiplicity = 2
        else:
            multiplicity = self.multiplicity - 1

        # Get charges of the radical cation (n-1 state)
        feature_value_n_minus1, error_message_n_minus1 = self._run_calculation(
            OMP_STACKSIZE=self.OMP_STACKSIZE,
            NUM_THREADS=self.NUM_THREADS,
            charge_scheme=self.iterable_option,
            feature_cache=self.feature_cache_n_minus1,
            feature_dimensionality=self.feature_dimensionality,
            ensemble_dimensionality=self.ensemble_dimensionality,
            conformer_idx=self.conformer_idx,
            conformer_name=f"{self.conformer_name}__n-1",
            electronic_struc_n=self.electronic_struc_n_minus1,  # pass electronic structure data for n-1 state to Multiwfn
            feature_type=self.feature_type,
            mol=self.mol,
            multiplicity=multiplicity,
            atom_bond_idx=self.atom_bond_idx,
            _keep_output_files=self._keep_output_files,
        )

        if error_message_n_minus1 is not None:
            self._err = error_message_n_minus1
            return

        # Calculate desired value and save it to the results dictionary
        assert isinstance(feature_value_n_minus1, (int, float))  # for type checker
        assert isinstance(feature_value_n_plus1, (int, float))  # for type checker
        assert isinstance(feature_value_n, (int, float))  # for type checker

        fukui_dual = round(
            number=2 * feature_value_n - feature_value_n_plus1 - feature_value_n_minus1, ndigits=6
        )
        self.results[self.atom_bond_idx] = {self.feature_name: fukui_dual}


class Multiwfn3DAtomCdftCondensedFukuiMinus(_Multiwfn3DAtomCdftCondensedFukui):
    """Feature factory for the 3D atom feature "cdft_condensed_fukui_minus", calculated with
    multiwfn.

    The index of this feature is 201 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_condensed_fukui_minus`` feature."""
        # Check if all electronic structure data is available
        self._check_electronic_structure_data(
            el_struc_data=self.electronic_struc_n_minus1,
            data_name="actual molecule minus one electron",
        )
        if self._err is not None:
            return

        # Modify feature_name to also include the name of the charge scheme
        self.feature_name = f"{self.feature_name}__{self.iterable_option}"

        # Get charges of the actual molecule
        feature_value_n, error_message_n = self._run_calculation(
            OMP_STACKSIZE=self.OMP_STACKSIZE,
            NUM_THREADS=self.NUM_THREADS,
            charge_scheme=self.iterable_option,
            feature_cache=self.feature_cache,
            feature_dimensionality=self.feature_dimensionality,
            ensemble_dimensionality=self.ensemble_dimensionality,
            conformer_idx=self.conformer_idx,
            conformer_name=self.conformer_name,
            electronic_struc_n=self.electronic_struc_n,
            feature_type=self.feature_type,
            mol=self.mol,
            multiplicity=self.multiplicity,
            atom_bond_idx=self.atom_bond_idx,
            _keep_output_files=self._keep_output_files,
        )

        if error_message_n is not None:
            self._err = error_message_n
            return

        # Calculate multiplicity of radical cation
        if self.multiplicity == 1:
            multiplicity = 2
        else:
            multiplicity = self.multiplicity - 1

        # Get charges of the radical cation (n-1 state)
        feature_value_n_minus1, error_message_n_minus1 = self._run_calculation(
            OMP_STACKSIZE=self.OMP_STACKSIZE,
            NUM_THREADS=self.NUM_THREADS,
            charge_scheme=self.iterable_option,
            feature_cache=self.feature_cache_n_minus1,
            feature_dimensionality=self.feature_dimensionality,
            ensemble_dimensionality=self.ensemble_dimensionality,
            conformer_idx=self.conformer_idx,
            conformer_name=f"{self.conformer_name}__n-1",
            electronic_struc_n=self.electronic_struc_n_minus1,  # pass electronic structure data for n-1 state to Multiwfn
            feature_type=self.feature_type,
            mol=self.mol,
            multiplicity=multiplicity,
            atom_bond_idx=self.atom_bond_idx,
            _keep_output_files=self._keep_output_files,
        )

        if error_message_n_minus1 is not None:
            self._err = error_message_n_minus1
            return

        # Calculate desired value and save it to the results dictionary
        assert isinstance(feature_value_n_minus1, (int, float))  # for type checker
        assert isinstance(feature_value_n, (int, float))  # for type checker

        fukui_minus = round(number=feature_value_n_minus1 - feature_value_n, ndigits=6)
        self.results[self.atom_bond_idx] = {self.feature_name: fukui_minus}


class Multiwfn3DAtomCdftCondensedFukuiPlus(_Multiwfn3DAtomCdftCondensedFukui):
    """Feature factory for the 3D atom feature "cdft_condensed_fukui_plus", calculated with
    multiwfn.

    The index of this feature is 202 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_condensed_fukui_plus`` feature."""
        # Check if the second electronic structure data is available
        self._check_electronic_structure_data(
            el_struc_data=self.electronic_struc_n_plus1,
            data_name="actual molecule plus one electron",
        )
        if self._err is not None:
            return

        # Modify feature_name to also include the name of the charge scheme
        self.feature_name = f"{self.feature_name}__{self.iterable_option}"

        # Get charges of the actual molecule
        feature_value_n, error_message_n = self._run_calculation(
            OMP_STACKSIZE=self.OMP_STACKSIZE,
            NUM_THREADS=self.NUM_THREADS,
            charge_scheme=self.iterable_option,
            feature_cache=self.feature_cache,
            feature_dimensionality=self.feature_dimensionality,
            ensemble_dimensionality=self.ensemble_dimensionality,
            conformer_idx=self.conformer_idx,
            conformer_name=self.conformer_name,
            electronic_struc_n=self.electronic_struc_n,
            feature_type=self.feature_type,
            mol=self.mol,
            multiplicity=self.multiplicity,
            atom_bond_idx=self.atom_bond_idx,
            _keep_output_files=self._keep_output_files,
        )

        if error_message_n is not None:
            self._err = error_message_n
            return

        # Calculate multiplicity of radical anion
        if self.multiplicity == 1:
            multiplicity = 2
        else:
            multiplicity = self.multiplicity - 1

        # Get charges of the radical anion (n+1 state)
        feature_value_n_plus1, error_message_n_plus1 = self._run_calculation(
            OMP_STACKSIZE=self.OMP_STACKSIZE,
            NUM_THREADS=self.NUM_THREADS,
            charge_scheme=self.iterable_option,
            feature_cache=self.feature_cache_n_plus1,
            feature_dimensionality=self.feature_dimensionality,
            ensemble_dimensionality=self.ensemble_dimensionality,
            conformer_idx=self.conformer_idx,
            conformer_name=f"{self.conformer_name}__n+1",
            electronic_struc_n=self.electronic_struc_n_plus1,  # pass electronic structure data for n+1 state to Multiwfn
            feature_type=self.feature_type,
            mol=self.mol,
            multiplicity=multiplicity,
            atom_bond_idx=self.atom_bond_idx,
            _keep_output_files=self._keep_output_files,
        )

        if error_message_n_plus1 is not None:
            self._err = error_message_n_plus1
            return

        # Calculate desired value and save it to the results dictionary
        assert isinstance(feature_value_n, (int, float))  # for type checker
        assert isinstance(feature_value_n_plus1, (int, float))  # for type checker

        fukui_plus = round(number=feature_value_n - feature_value_n_plus1, ndigits=6)
        self.results[self.atom_bond_idx] = {self.feature_name: fukui_plus}


class Multiwfn3DAtomCdftCondensedFukuiZero(_Multiwfn3DAtomCdftCondensedFukui):
    """Feature factory for the 3D atom feature "cdft_condensed_fukui_zero", calculated with
    multiwfn.

    The index of this feature is 203 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_condensed_fukui_zero`` feature."""
        # Check if both (radical anion and cation) electronic structure data is available
        self._check_electronic_structure_data(
            el_struc_data=self.electronic_struc_n_plus1,
            data_name="actual molecule plus one electron",
        )
        if self._err is not None:
            return

        self._check_electronic_structure_data(
            el_struc_data=self.electronic_struc_n_minus1,
            data_name="actual molecule minus one electron",
        )
        if self._err is not None:
            return

        # Modify feature_name to also include the name of the charge scheme
        self.feature_name = f"{self.feature_name}__{self.iterable_option}"

        # Calculate multiplicity of radical anion
        if self.multiplicity == 1:
            multiplicity = 2
        else:
            multiplicity = self.multiplicity - 1

        # Get charges of the radical anion (n+1 state)
        feature_value_n_plus1, error_message_n_plus1 = self._run_calculation(
            OMP_STACKSIZE=self.OMP_STACKSIZE,
            NUM_THREADS=self.NUM_THREADS,
            charge_scheme=self.iterable_option,
            feature_cache=self.feature_cache_n_plus1,
            feature_dimensionality=self.feature_dimensionality,
            ensemble_dimensionality=self.ensemble_dimensionality,
            conformer_idx=self.conformer_idx,
            conformer_name=f"{self.conformer_name}__n+1",
            electronic_struc_n=self.electronic_struc_n_plus1,  # pass electronic structure data for n+1 state to Multiwfn
            feature_type=self.feature_type,
            mol=self.mol,
            multiplicity=multiplicity,
            atom_bond_idx=self.atom_bond_idx,
            _keep_output_files=self._keep_output_files,
        )

        if error_message_n_plus1 is not None:
            self._err = error_message_n_plus1
            return

        # Calculate multiplicity of radical cation
        if self.multiplicity == 1:
            multiplicity = 2
        else:
            multiplicity = self.multiplicity - 1

        # Get charges of the radical cation (n-1 state)
        feature_value_n_minus1, error_message_n_minus1 = self._run_calculation(
            OMP_STACKSIZE=self.OMP_STACKSIZE,
            NUM_THREADS=self.NUM_THREADS,
            charge_scheme=self.iterable_option,
            feature_cache=self.feature_cache_n_minus1,
            feature_dimensionality=self.feature_dimensionality,
            ensemble_dimensionality=self.ensemble_dimensionality,
            conformer_idx=self.conformer_idx,
            conformer_name=f"{self.conformer_name}__n-1",
            electronic_struc_n=self.electronic_struc_n_minus1,  # pass electronic structure data for n-1 state to Multiwfn
            feature_type=self.feature_type,
            mol=self.mol,
            multiplicity=multiplicity,
            atom_bond_idx=self.atom_bond_idx,
            _keep_output_files=self._keep_output_files,
        )

        if error_message_n_minus1 is not None:
            self._err = error_message_n_minus1
            return

        # Calculate desired value and save it to the results dictionary
        assert isinstance(feature_value_n_minus1, (int, float))  # for type checker
        assert isinstance(feature_value_n_plus1, (int, float))  # for type checker

        fukui_zero = round(number=(feature_value_n_minus1 - feature_value_n_plus1) / 2, ndigits=6)
        self.results[self.atom_bond_idx] = {self.feature_name: fukui_zero}
