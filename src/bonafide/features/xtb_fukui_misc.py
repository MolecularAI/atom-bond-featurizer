"""Condensed Fukui coefficients and some additional features from the semi-empirical quantum
chemistry package xtb.
"""

import os
from typing import Optional

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.global_properties import (
    calculate_global_cdft_descriptors_fmo,
)
from bonafide.utils.sp_xtb import XtbSP


class _Xtb3DAtomFukuiMisc(BaseFeaturizer):
    """Parent feature factory for the 3D atom Fukui and miscellaneous features calculated
    with xtb.
    """

    method: str

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def _get_fukuis(self) -> None:
        """Calculate the condensed Fukui indices using xtb and read them from the output file.

        Returns
        -------
        None
        """
        out_file_prefix = "Xtb3DAtomCdftCondensedFukui"
        self._run_xtb(calc_fukui=True, calc_ceh=False, out_file_prefix=out_file_prefix)
        self._read_output_file(enforce="fukui", out_file_prefix=out_file_prefix)

    def _run_xtb(self, calc_fukui: bool, calc_ceh: bool, out_file_prefix: str) -> None:
        """Run an xtb single-point energy calculation.

        Parameters
        ----------
        calc_fukui : bool
            Whether to calculate Fukui indices.
        calc_ceh : bool
            Whether to calculate Hueckel charges.
        out_file_prefix : str
            The prefix for the output file name.

        Returns
        -------
        None
        """
        params = dict(vars(self))

        # Overwrite the etemp parameter with the etemp_native parameter
        params["etemp"] = params["etemp_native"]

        # Run xtb
        sp = XtbSP(**params)
        sp.calculate(
            write_el_struc_file=False,
            calc_fukui=calc_fukui,
            calc_ceh=calc_ceh,
            out_file_name=f"{out_file_prefix}__{self.conformer_name}",
        )

    def _read_output_file(self, enforce: str, out_file_prefix: str) -> None:
        """Read the output file, extract the respective data and write it to ``results``.

        This method is used to extract Fukui indices or other atomic properties from an xtb output
        file.

        Parameters
        ----------
        enforce : str
            Whether to enforce extraction of "fukui" or "misc" data.
        out_file_prefix : str
            Prefix of the output file name.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"{out_file_prefix}__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Xtb output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            xtb_output = f.readlines()

        # Find relevant position in the file and check if it was found
        start_idx = None
        start_idx2 = None

        homo_energy = None
        lumo_energy = None

        for line_idx, line in enumerate(xtb_output):
            # Fukui indices
            if all(["#" in line, "f(+)" in line, "f(-)" in line, "f(0)" in line]):
                start_idx = line_idx + 1

            # Dispersion and polarizability
            if all(["#" in line, "Z" in line, "covCN" in line, "q" in line, "C6AA" in line]):
                start_idx2 = line_idx + 1

            # HOMO/LUMO (directly get values)
            if all([homo_energy is None and "(HOMO)" in line]):
                homo_energy = float(line.split()[-2])
            if all([lumo_energy is None and "(LUMO)" in line]):
                lumo_energy = float(line.split()[-2])

        # Return if error occurred depending on what was requested
        _errmsg = (
            f"output file generated through '{self.__class__.__name__}' does not contain "
            "the requested data; probably the calculation failed. Check the output file."
        )
        if homo_energy is None:
            self._err = _errmsg
            return
        if start_idx is None and enforce == "fukui":
            self._err = _errmsg
            return
        if start_idx2 is None and enforce == "misc":
            self._err = _errmsg
            return

        # Calculate global C-DFT descriptors (must be done here because FMO energies are only
        # available here)
        assert homo_energy is not None  # for type checker
        assert lumo_energy is not None  # for type checker

        error_message = self._calculate_global_descriptors_fmo(
            homo_energy=homo_energy, lumo_energy=lumo_energy
        )
        if error_message is not None:
            self._err = error_message
            return

        # Extract Fukui data and write all available data to the results dictionary
        if enforce == "fukui":
            for line_idx, line in enumerate(
                xtb_output[start_idx : start_idx + self.mol.GetNumAtoms()]
            ):
                splitted = line.strip().split()
                fukui_plus = float(splitted[-3])
                fukui_minus = float(splitted[-2])
                fukui_zero = float(splitted[-1])
                fukui_dual = round((fukui_plus - fukui_minus), 6)

                self.results[line_idx] = {
                    "xtb3D-atom-cdft_condensed_fukui_plus": fukui_plus,
                    "xtb3D-atom-cdft_condensed_fukui_minus": fukui_minus,
                    "xtb3D-atom-cdft_condensed_fukui_zero": fukui_zero,
                    "xtb3D-atom-cdft_condensed_fukui_dual": fukui_dual,
                }

        # Extract misc data if requested or optionally if available but not extracted yet
        get_misc = False
        if enforce == "misc":
            get_misc = True
        elif 0 in self.results:
            if all(
                [
                    start_idx2 is not None,
                    "xtb3D-atom-c6_dispersion_coefficient" not in self.results[0],
                ]
            ):
                get_misc = True

        if get_misc is True:
            for line_idx, line in enumerate(
                xtb_output[start_idx2 : start_idx2 + self.mol.GetNumAtoms()]
            ):
                splitted = line.strip().split()
                disp = float(splitted[-2])
                pol = float(splitted[-1])

                if line_idx not in self.results:
                    self.results[line_idx] = {}
                self.results[line_idx]["xtb3D-atom-c6_dispersion_coefficient"] = disp
                self.results[line_idx]["xtb3D-atom-polarizability"] = pol

    def _read_output_file2(self) -> None:
        """Read the output file, extract the respective data and write it to ``results``.

        This method is used to extract Hueckel charges from an xtb ceh.charges file.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = "ceh.charges"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Xtb output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            xtb_output = f.readlines()

        # Read data and write it to the results dictionary
        for line_idx, value in enumerate(xtb_output):
            self.results[line_idx] = {self.feature_name: float(value.strip())}

        # Rename the ceh.charges file
        os.rename(_opath, f"{self.__class__.__name__}__{self.conformer_name}.charges")

    def _calculate_global_descriptors_fmo(
        self, homo_energy: float, lumo_energy: float
    ) -> Optional[str]:
        """Calculate molecule-level descriptors from the HOMO and LUMO energy.

        The included descriptors are:

        * HOMO-LUMO gap
        * Chemical potential
        * Hardness
        * Softness
        * Electrophilicity
        * Nucleophilicity

        Parameters
        ----------
        homo_energy : float
            The energy of the highest occupied molecular orbital of the molecule in eV.
        lumo_energy : float
            The energy of the lowest unoccupied molecular orbital of the molecule in eV.

        Returns
        -------
        Optional[str]
            Returns an error message if the calculation of the C-DFT descriptors failed or
            ``None`` if everything worked as expected.
        """
        # Check if the global descriptors have already been calculated
        if "xtb3D-global-homo_energy" in self.global_feature_cache[self.conformer_idx]:
            return None

        # Get the C-DFT descriptors
        (
            error_message,
            homo_lumo_gap,
            chem_potential,
            hardness,
            softness,
            electrophilicity,
            nucleophilicity,
        ) = calculate_global_cdft_descriptors_fmo(homo_energy=homo_energy, lumo_energy=lumo_energy)
        if error_message is not None:
            return error_message

        # Write the data to the global feature cache
        self.global_feature_cache[self.conformer_idx]["xtb3D-global-homo_energy"] = homo_energy
        self.global_feature_cache[self.conformer_idx]["xtb3D-global-lumo_energy"] = lumo_energy
        self.global_feature_cache[self.conformer_idx]["xtb3D-global-homo_lumo_gap"] = homo_lumo_gap
        self.global_feature_cache[self.conformer_idx]["xtb3D-global-chem_potential_fmo"] = (
            chem_potential
        )
        self.global_feature_cache[self.conformer_idx]["xtb3D-global-hardness_fmo"] = hardness
        self.global_feature_cache[self.conformer_idx]["xtb3D-global-softness_fmo"] = softness
        self.global_feature_cache[self.conformer_idx]["xtb3D-global-electrophilicity_fmo"] = (
            electrophilicity
        )
        self.global_feature_cache[self.conformer_idx]["xtb3D-global-nucleophilicity_fmo"] = (
            nucleophilicity
        )

        return None

    def _check_xtb_method(self, allowed_method: str) -> None:
        """Check if the selected xtb method is allowed for the specific feature.

        Parameters
        ----------
        allowed_method : str
            The name of the allowed xtb method.

        Returns
        -------
        None
        """
        if self.method != allowed_method:
            self._err = (
                f"the '{self.feature_name}' feature cannot be calculated with the "
                f"'{self.method}' method; use '{allowed_method}' instead"
            )


class Xtb3DAtomC6DispersionCoefficient(_Xtb3DAtomFukuiMisc):
    """Feature factory for the 3D atom feature "c6_dispersion_coefficient", calculated with xtb.

    The index of this feature is 561 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-c6_dispersion_coefficient`` feature."""
        self._check_xtb_method(allowed_method="gfn2-xtb")
        if self._err is not None:
            return

        out_file_prefix = "Xtb3DAtomDescriptors"
        self._run_xtb(calc_fukui=False, calc_ceh=False, out_file_prefix=out_file_prefix)
        self._read_output_file(enforce="misc", out_file_prefix=out_file_prefix)


class Xtb3DAtomCdftCondensedFukuiDual(_Xtb3DAtomFukuiMisc):
    """Feature factory for the 3D atom feature "cdft_condensed_fukui_dual", calculated with xtb.

    The index of this feature is 562 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_condensed_fukui_dual`` feature."""
        self._get_fukuis()


class Xtb3DAtomCdftCondensedFukuiMinus(_Xtb3DAtomFukuiMisc):
    """Feature factory for the 3D atom feature "cdft_condensed_fukui_minus", calculated with
    xtb.

    The index of this feature is 563 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_condensed_fukui_minus`` feature."""
        self._get_fukuis()


class Xtb3DAtomCdftCondensedFukuiPlus(_Xtb3DAtomFukuiMisc):
    """Feature factory for the 3D atom feature "cdft_condensed_fukui_plus", calculated with xtb.

    The index of this feature is 564 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_condensed_fukui_plus`` feature."""
        self._get_fukuis()


class Xtb3DAtomCdftCondensedFukuiZero(_Xtb3DAtomFukuiMisc):
    """Feature factory for the 3D atom feature "cdft_condensed_fukui_zero", calculated with xtb.

    The index of this feature is 565 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_condensed_fukui_zero`` feature."""
        self._get_fukuis()


class Xtb3DAtomPartialChargeHueckel(_Xtb3DAtomFukuiMisc):
    """Feature factory for the 3D atom feature "partial_charge_hueckel", calculated with xtb.

    The index of this feature is 588 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-partial_charge_hueckel`` feature."""
        out_file_prefix = "Xtb3DAtomPartialChargeHueckel"
        self._run_xtb(calc_fukui=False, calc_ceh=True, out_file_prefix=out_file_prefix)
        self._read_output_file2()


class Xtb3DAtomPolarizability(_Xtb3DAtomFukuiMisc):
    """Feature factory for the 3D atom feature "polarizability", calculated with xtb.

    The index of this feature is 589 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-polarizability`` feature."""
        self._check_xtb_method(allowed_method="gfn2-xtb")
        if self._err is not None:
            return

        out_file_prefix = "Xtb3DAtomDescriptors"
        self._run_xtb(calc_fukui=False, calc_ceh=False, out_file_prefix=out_file_prefix)
        self._read_output_file(enforce="misc", out_file_prefix=out_file_prefix)
