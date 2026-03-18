"""Hilbert space features from ``Multiwfn``."""

import os
from typing import List, Union

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver


class _Multiwfn3DHilbertSpace(BaseFeaturizer):
    """Parent feature factory for the 3D atom and bond Multiwfn Hilbert space features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def _run_multiwfn(self, feature_type: str) -> None:
        """Run Multiwfn.

        Parameters
        ----------
        feature_type : str
            The type of the feature to calculate, either "atom" or "bond".

        Returns
        -------
        None
        """
        # Select Hilbert space analysis
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [200, 2]

        # Run analysis and exit program
        if feature_type == "atom":
            multiwfn_commands.extend([1, -1])
            multiwfn_commands.extend([0, 0, 0, "q"])
            output_file_name = f"Multiwfn3DAtomHilbertSpace__{self.conformer_name}"

        if feature_type == "bond":
            multiwfn_commands.extend([2, "b"])
            multiwfn_commands.extend(["q", 0, 0, "q"])
            output_file_name = f"Multiwfn3DBondHilbertSpace__{self.conformer_name}"

        # Set up environment variables
        environment_variables = {
            var: getattr(self, var, None) for var in PROGRAM_ENVIRONMENT_VARIABLES["multiwfn"]
        }

        # Run Multiwfn
        multiwfn_driver(
            cmds=multiwfn_commands,
            input_file_path=str(self.electronic_struc_n),
            output_file_name=output_file_name,
            environment_variables=environment_variables,
            namespace=self.conformer_name[::-1].split("__", 1)[-1][::-1],
        )

    def _read_output_file_atom(self) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used to process the atom features.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"Multiwfn3DAtomHilbertSpace__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Extract data and write it to the results dictionary
        for line_idx, line in enumerate(multiwfn_output):
            if line.startswith(" Result of atom"):
                atom_idx = int(line.split("atom")[-1].split("(")[0])
                pop_number = float(multiwfn_output[line_idx + 1].split(":")[-1])

                _x_, _y_, _z_, dipole_norm_ = multiwfn_output[line_idx + 3].split("=")[1:]
                _x = float(_x_.split()[0])
                _y = float(_y_.split()[0])
                _z = float(_z_.split()[0])
                dipole_norm = float(dipole_norm_)
                dipole = ",".join([str(_x), str(_y), str(_z)])

                _x_, _y_, _z_, contrib_nuc_norm_ = multiwfn_output[line_idx + 5].split("=")[1:]
                _x = float(_x_.split()[0])
                _y = float(_y_.split()[0])
                _z = float(_z_.split()[0])
                contrib_nuc_norm = float(contrib_nuc_norm_)
                contrib_nuc_dipole = ",".join([str(_x), str(_y), str(_z)])

                _x_, _y_, _z_, contrib_el_norm_ = multiwfn_output[line_idx + 7].split("=")[1:]
                _x = float(_x_.split()[0])
                _y = float(_y_.split()[0])
                _z = float(_z_.split()[0])
                contrib_el_norm = float(contrib_el_norm_)
                contrib_el_dipole = ",".join([str(_x), str(_y), str(_z)])

                _x_, _y_, _z_, contrib_norm_ = multiwfn_output[line_idx + 9].split("=")[1:]
                _x = float(_x_.split()[0])
                _y = float(_y_.split()[0])
                _z = float(_z_.split()[0])
                contrib_norm = float(contrib_norm_)
                contrib_dipole = ",".join([str(_x), str(_y), str(_z)])

                self.results[atom_idx - 1] = {
                    "multiwfn3D-atom-hilbert_space_local_population_number": pop_number,
                    "multiwfn3D-atom-hilbert_space_dipole_moment": dipole,
                    "multiwfn3D-atom-hilbert_space_dipole_moment_norm": dipole_norm,
                    "multiwfn3D-atom-hilbert_space_contribution_to_system_dipole_moment_nuclear_charge": contrib_nuc_dipole,
                    "multiwfn3D-atom-hilbert_space_contribution_to_system_dipole_moment_nuclear_charge_norm": contrib_nuc_norm,
                    "multiwfn3D-atom-hilbert_space_contribution_to_system_dipole_moment_electrons": contrib_el_dipole,
                    "multiwfn3D-atom-hilbert_space_contribution_to_system_dipole_moment_electrons_norm": contrib_el_norm,
                    "multiwfn3D-atom-hilbert_space_contribution_to_system_dipole_moment": contrib_dipole,
                    "multiwfn3D-atom-hilbert_space_contribution_to_system_dipole_moment_norm": contrib_norm,
                }

    def _read_output_file_bond(self) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used to process the bond features.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"Multiwfn3DBondHilbertSpace__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Extract data and write it to the results dictionary
        for line_idx, line in enumerate(multiwfn_output):
            if all(["Result between atom" in line, "and atom" in line, "distance:" in line]):
                atom_idx_1_, atom_idx_2_ = line.split("atom")[1:]
                atom_idx_1 = int(atom_idx_1_.split("(")[0]) - 1
                atom_idx_2 = int(atom_idx_2_.split("(")[0]) - 1

                pop_number = float(multiwfn_output[line_idx + 1].split(":")[-1])

                _x_, _y_, _z_, dipole_norm_ = multiwfn_output[line_idx + 3].split("=")[1:]
                _x = float(_x_.split()[0])
                _y = float(_y_.split()[0])
                _z = float(_z_.split()[0])
                dipole_norm = float(dipole_norm_)
                dipole = ",".join([str(_x), str(_y), str(_z)])

                _x_, _y_, _z_, contrib_norm_ = multiwfn_output[line_idx + 5].split("=")[1:]
                _x = float(_x_.split()[0])
                _y = float(_y_.split()[0])
                _z = float(_z_.split()[0])
                contrib_norm = float(contrib_norm_)
                contrib_dipole = ",".join([str(_x), str(_y), str(_z)])

                # Find the bond and write the data to the results dictionary
                for bond in self.mol.GetBonds():
                    bond_idx = bond.GetIdx()
                    if any(
                        [
                            (
                                bond.GetBeginAtomIdx() == atom_idx_1
                                and bond.GetEndAtomIdx() == atom_idx_2
                            ),
                            (
                                bond.GetBeginAtomIdx() == atom_idx_2
                                and bond.GetEndAtomIdx() == atom_idx_1
                            ),
                        ]
                    ):
                        self.results[bond_idx] = {
                            "multiwfn3D-bond-hilbert_space_overlap_population": pop_number,
                            "multiwfn3D-bond-hilbert_space_dipole_moment": dipole,
                            "multiwfn3D-bond-hilbert_space_dipole_moment_norm": dipole_norm,
                            "multiwfn3D-bond-hilbert_space_contribution_to_system_dipole_moment": contrib_dipole,
                            "multiwfn3D-bond-hilbert_space_contribution_to_system_dipole_moment_norm": contrib_norm,
                        }
                        break


class Multiwfn3DAtomHilbertSpaceContributionToSystemDipoleMoment(_Multiwfn3DHilbertSpace):
    """Feature factory for the 3D atom feature
    "hilbert_space_contribution_to_system_dipole_moment", calculated with multiwfn.

    The index of this feature is 255 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-hilbert_space_contribution_to_system_dipole_moment``
        feature."""
        self._run_multiwfn(feature_type="atom")
        self._read_output_file_atom()


class Multiwfn3DAtomHilbertSpaceContributionToSystemDipoleMomentElectrons(_Multiwfn3DHilbertSpace):
    """Feature factory for the 3D atom feature
    "hilbert_space_contribution_to_system_dipole_moment_electrons", calculated with multiwfn.

    The index of this feature is 256 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-hilbert_space_contribution_to_system_dipole_moment_electrons``
        feature."""
        self._run_multiwfn(feature_type="atom")
        self._read_output_file_atom()


class Multiwfn3DAtomHilbertSpaceContributionToSystemDipoleMomentElectronsNorm(
    _Multiwfn3DHilbertSpace
):
    """Feature factory for the 3D atom feature
    "hilbert_space_contribution_to_system_dipole_moment_electrons_norm", calculated with
    multiwfn.

    The index of this feature is 257 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-hilbert_space_contribution_to_system_dipole_moment_electrons_norm``
        feature."""
        self._run_multiwfn(feature_type="atom")
        self._read_output_file_atom()


class Multiwfn3DAtomHilbertSpaceContributionToSystemDipoleMomentNorm(_Multiwfn3DHilbertSpace):
    """Feature factory for the 3D atom feature
    "hilbert_space_contribution_to_system_dipole_moment_norm", calculated with multiwfn.

    The index of this feature is 258 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-hilbert_space_contribution_to_system_dipole_moment_norm`` feature."""
        self._run_multiwfn(feature_type="atom")
        self._read_output_file_atom()


class Multiwfn3DAtomHilbertSpaceContributionToSystemDipoleMomentNuclearCharge(
    _Multiwfn3DHilbertSpace
):
    """Feature factory for the 3D atom feature
    "hilbert_space_contribution_to_system_dipole_moment_nuclear_charge", calculated with
    multiwfn.

    The index of this feature is 259 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-hilbert_space_contribution_to_system_dipole_moment_nuclear_charge``
        feature."""
        self._run_multiwfn(feature_type="atom")
        self._read_output_file_atom()


class Multiwfn3DAtomHilbertSpaceContributionToSystemDipoleMomentNuclearChargeNorm(
    _Multiwfn3DHilbertSpace
):
    """Feature factory for the 3D atom feature
    "hilbert_space_contribution_to_system_dipole_moment_nuclear_charge_norm", calculated with
    multiwfn.

    The index of this feature is 260 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-hilbert_space_contribution_to_system_dipole_moment_nuclear_charge_norm``
        feature."""
        self._run_multiwfn(feature_type="atom")
        self._read_output_file_atom()


class Multiwfn3DAtomHilbertSpaceDipoleMoment(_Multiwfn3DHilbertSpace):
    """Feature factory for the 3D atom feature "hilbert_space_dipole_moment", calculated with
    multiwfn.

    The index of this feature is 261 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-hilbert_space_dipole_moment`` feature."""
        self._run_multiwfn(feature_type="atom")
        self._read_output_file_atom()


class Multiwfn3DAtomHilbertSpaceDipoleMomentNorm(_Multiwfn3DHilbertSpace):
    """Feature factory for the 3D atom feature "hilbert_space_dipole_moment_norm", calculated
    with multiwfn.

    The index of this feature is 262 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-hilbert_space_dipole_moment_norm`` feature."""
        self._run_multiwfn(feature_type="atom")
        self._read_output_file_atom()


class Multiwfn3DAtomHilbertSpaceLocalPopulationNumber(_Multiwfn3DHilbertSpace):
    """Feature factory for the 3D atom feature "hilbert_space_local_population_number",
    calculated with multiwfn.

    The index of this feature is 263 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-hilbert_space_local_population_number`` feature."""
        self._run_multiwfn(feature_type="atom")
        self._read_output_file_atom()


class Multiwfn3DBondHilbertSpaceContributionToSystemDipoleMoment(_Multiwfn3DHilbertSpace):
    """Feature factory for the 3D bond feature
    "hilbert_space_contribution_to_system_dipole_moment", calculated with multiwfn.

    The index of this feature is 440 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-hilbert_space_contribution_to_system_dipole_moment``
        feature."""
        self._run_multiwfn(feature_type="bond")
        self._read_output_file_bond()


class Multiwfn3DBondHilbertSpaceContributionToSystemDipoleMomentNorm(_Multiwfn3DHilbertSpace):
    """Feature factory for the 3D bond feature
    "hilbert_space_contribution_to_system_dipole_moment_norm", calculated with multiwfn.

    The index of this feature is 441 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-bond-hilbert_space_contribution_to_system_dipole_moment_norm`` feature."""
        self._run_multiwfn(feature_type="bond")
        self._read_output_file_bond()


class Multiwfn3DBondHilbertSpaceDipoleMoment(_Multiwfn3DHilbertSpace):
    """Feature factory for the 3D bond feature "hilbert_space_dipole_moment", calculated with
    multiwfn.

    The index of this feature is 442 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-hilbert_space_dipole_moment`` feature."""
        self._run_multiwfn(feature_type="bond")
        self._read_output_file_bond()


class Multiwfn3DBondHilbertSpaceDipoleMomentNorm(_Multiwfn3DHilbertSpace):
    """Feature factory for the 3D bond feature "hilbert_space_dipole_moment_norm", calculated
    with multiwfn.

    The index of this feature is 443 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-hilbert_space_dipole_moment_norm`` feature."""
        self._run_multiwfn(feature_type="bond")
        self._read_output_file_bond()


class Multiwfn3DBondHilbertSpaceOverlapPopulation(_Multiwfn3DHilbertSpace):
    """Feature factory for the 3D bond feature "hilbert_space_overlap_population", calculated
    with multiwfn.

    The index of this feature is 444 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-bond-hilbert_space_overlap_population`` feature."""
        self._run_multiwfn(feature_type="bond")
        self._read_output_file_bond()
