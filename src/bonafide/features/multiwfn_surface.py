"""Molecular surface features from ``Multiwfn``."""

import math
import os
from typing import List, Union

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver


class _Multiwfn3DAtomSurface(BaseFeaturizer):
    """Parent feature factory for the 3D atom surface Multiwfn features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    grid_point_spacing: float
    surface_definition: int
    surface_iso_value: float

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def _run_multiwfn(self, mapping_function_idx: Union[int, str], id_string: str) -> None:
        """Run Multiwfn.

        Parameters
        ----------
        mapping_function_idx : Union[int, str]
            The Multiwfn option(s) to select the real space function requested by the user along
            with additional input if required.
        id_string : str
            A feature-specific identifier string to distinguish the output files of different
            features.

        Returns
        -------
        None
        """
        # Select surface analysis
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [12]

        # Add grid point spacing to command
        multiwfn_commands.extend([3, self.grid_point_spacing])

        # Add surface definition to command
        if self.surface_definition == 1:
            multiwfn_commands.extend([1, self.surface_definition, self.surface_iso_value])
        else:
            multiwfn_commands.extend([1, 2, self.surface_definition, self.surface_iso_value])

        # Add mapping function to command
        multiwfn_commands.extend([2, mapping_function_idx])

        # Exit program
        multiwfn_commands.extend([0, 11, "n", -1, -1, "q"])

        # Set up environment variables
        environment_variables = {
            var: getattr(self, var, None) for var in PROGRAM_ENVIRONMENT_VARIABLES["multiwfn"]
        }

        # Run Multiwfn
        multiwfn_driver(
            cmds=multiwfn_commands,
            input_file_path=str(self.electronic_struc_n),
            output_file_name=f"Multiwfn3DAtomSurface{id_string}__{self.conformer_name}",
            environment_variables=environment_variables,
            namespace=self.conformer_name[::-1].split("__", 1)[-1][::-1],
        )

    def _read_output_file(self, id_string: str, feature_name: str) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used to read the data for the average local ionization energy features.

        Parameters
        ----------
        id_string : str
            A feature-specific identifier string to distinguish the output files of different
            features.
        feature_name : str
            The name of the feature.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"Multiwfn3DAtomSurface{id_string}__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Extract data
        start_idx = None
        for line_idx, line in enumerate(multiwfn_output):
            if all(
                [
                    "Atom#" in line,
                    "Area(Ang^2)" in line,
                    "Min value" in line,
                    "Max value" in line,
                    "Average" in line,
                    "Variance" in line,
                ]
            ):
                start_idx = line_idx + 1

        # Check if start_idx was found
        if start_idx is None:
            self._err = (
                f"output file generated through 'Multiwfn3DAtomSurface{id_string}' does "
                "not contain the requested data; probably the calculation failed. Check the "
                "output file"
            )
            return

        # Save values to results dictionary
        for line in multiwfn_output[start_idx:]:
            if line.strip() == "":
                break

            splitted = line.split()
            min_val = float(splitted[2])
            max_val = float(splitted[3])
            avg_val = float(splitted[4])

            self.results[int(splitted[0]) - 1] = {
                f"multiwfn3D-atom-surface_{feature_name}_min": min_val,
                f"multiwfn3D-atom-surface_{feature_name}_max": max_val,
                f"multiwfn3D-atom-surface_{feature_name}_mean": avg_val,
            }

    def _read_output_file2(self, id_string: str, feature_name: str) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used to read the data for all surface features except for the average local
        ionization energy and the electrostatic potential features.

        Parameters
        ----------
        id_string : str
            A feature-specific identifier string to distinguish the output files of different
            features.
        feature_name : str
            The name of the feature.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"Multiwfn3DAtomSurface{id_string}__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Extract data
        start_idx_1 = None
        start_idx_2 = None
        for line_idx, line in enumerate(multiwfn_output):
            # Start index 1
            if all(
                [
                    "Atom#" in line,
                    "All/Positive/Negative area (Ang^2)" in line,
                    "Minimal value" in line,
                    "Maximal value" in line,
                ]
            ):
                start_idx_1 = line_idx + 1

            # Start index 2
            if all(
                [
                    "Atom#" in line,
                    "All/Positive/Negative average" in line,
                    "All/Positive/Negative variance" in line,
                ]
            ):
                start_idx_2 = line_idx + 1

        # Check if start indices were found
        if any([start_idx_1 is None, start_idx_2 is None]):
            self._err = (
                f"output file generated through 'Multiwfn3DAtomSurface{id_string}' does "
                "not contain the requested data; probably the calculation failed. Check the "
                "output file"
            )
            return

        # Save min and max values to results dictionary
        for line in multiwfn_output[start_idx_1:]:
            if line.strip() == "":
                break

            splitted = line.split()
            min_val = float(splitted[-2])
            max_val = float(splitted[-1])

            self.results[int(splitted[0]) - 1] = {
                f"multiwfn3D-atom-surface_{feature_name}_min": min_val,
                f"multiwfn3D-atom-surface_{feature_name}_max": max_val,
            }

        # Save average values to results dictionary
        for line in multiwfn_output[start_idx_2:]:
            if line.strip() == "":
                break

            splitted = line.split()
            avg_val = float(splitted[1])

            self.results[int(splitted[0]) - 1][f"multiwfn3D-atom-surface_{feature_name}_mean"] = (
                avg_val
            )

    def _read_output_file3(self, id_string: str, feature_name: str) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        This method is used to read the data for the electrostatic potential features.

        Parameters
        ----------
        id_string : str
            A feature-specific identifier string to distinguish the output files of different
            features.
        feature_name : str
            The name of the feature.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"Multiwfn3DAtomSurface{id_string}__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Extract data
        start_idx_1 = None
        start_idx_2 = None
        start_idx_3 = None
        for line_idx, line in enumerate(multiwfn_output):
            # Start index 1
            if all(
                [
                    "Atom#" in line,
                    "All/Positive/Negative area (Ang^2)" in line,
                    "Minimal value" in line,
                    "Maximal value" in line,
                ]
            ):
                start_idx_1 = line_idx + 1

            # Start index 2
            if all(
                [
                    "Atom#" in line,
                    "All/Positive/Negative average" in line,
                    "All/Positive/Negative variance" in line,
                ]
            ):
                start_idx_2 = line_idx + 1

            # Start index 3
            if all(["Atom#" in line, "Pi" in line, "nu" in line, "nu*sigma^2" in line]):
                start_idx_3 = line_idx + 1

        # Check if start indices were found
        if any(
            [
                start_idx_1 is None,
                start_idx_2 is None,
                start_idx_3 is None,
            ]
        ):
            self._err = (
                f"output file generated through 'Multiwfn3DAtomSurface{id_string}' does "
                "not contain the requested data; probably the calculation failed. Check the "
                "output file"
            )
            return

        # Save min and max values to results dictionary
        for line in multiwfn_output[start_idx_1:]:
            if line.strip() == "":
                break

            splitted = line.split()
            min_val = float(splitted[-2])
            max_val = float(splitted[-1])

            self.results[int(splitted[0]) - 1] = {
                f"multiwfn3D-atom-surface_{feature_name}_min": min_val,
                f"multiwfn3D-atom-surface_{feature_name}_max": max_val,
            }

        # Save average values to results dictionary
        for line in multiwfn_output[start_idx_2:]:
            if line.strip() == "":
                break

            splitted = line.split()
            avg_val = float(splitted[1])

            self.results[int(splitted[0]) - 1][f"multiwfn3D-atom-surface_{feature_name}_mean"] = (
                avg_val
            )

        # Save pi, nu, and nu*sigma^2 values to results dictionary
        for line in multiwfn_output[start_idx_3:]:
            if line.strip() == "":
                break

            splitted = line.split()
            pi = float(splitted[-3])
            nu_ = float(splitted[-2])
            nu_mul_ = float(splitted[-1])

            # Make nan to _inaccessible (only possible for nu and nu_mul)
            nu: Union[float, str]
            if math.isnan(nu_) is True:
                nu = "_inaccessible"
            else:
                nu = nu_

            nu_mul: Union[float, str]
            if math.isnan(nu_mul_) is True:
                nu_mul = "_inaccessible"
            else:
                nu_mul = nu_mul_

            self.results[int(splitted[0]) - 1][
                f"multiwfn3D-atom-surface_{feature_name}_internal_charge_separation"
            ] = pi
            self.results[int(splitted[0]) - 1][
                f"multiwfn3D-atom-surface_{feature_name}_balance_of_charges"
            ] = nu
            self.results[int(splitted[0]) - 1][
                f"multiwfn3D-atom-surface_{feature_name}_balance_of_charges_times_variance"
            ] = nu_mul


class Multiwfn3DAtomSurfaceAverageLocalIonizationEnergyMax(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_average_local_ionization_energy_max",
    calculated with multiwfn.

    The index of this feature is 368 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_average_local_ionization_energy_max``
        feature."""
        id_string = "AverageLocalIonizationEnergy"
        feature_name = "average_local_ionization_energy"
        self._run_multiwfn(mapping_function_idx=2, id_string=id_string)
        self._read_output_file(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceAverageLocalIonizationEnergyMean(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_average_local_ionization_energy_mean",
    calculated with multiwfn.

    The index of this feature is 369 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_average_local_ionization_energy_mean``
        feature."""
        id_string = "AverageLocalIonizationEnergy"
        feature_name = "average_local_ionization_energy"
        self._run_multiwfn(mapping_function_idx=2, id_string=id_string)
        self._read_output_file(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceAverageLocalIonizationEnergyMin(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_average_local_ionization_energy_min",
    calculated with multiwfn.

    The index of this feature is 370 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_average_local_ionization_energy_min``
        feature."""
        id_string = "AverageLocalIonizationEnergy"
        feature_name = "average_local_ionization_energy"
        self._run_multiwfn(mapping_function_idx=2, id_string=id_string)
        self._read_output_file(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceElectronDelocalizationRangeFunctionMax(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature
    "surface_electron_delocalization_range_function_max", calculated with multiwfn.

    The index of this feature is 371 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    length_scale: float

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_electron_delocalization_range_function_max``
        feature."""
        id_string = "ElectronDelocalizationRangeFunction"
        feature_name = "electron_delocalization_range_function"

        # Add additional option to mapping_function_idx parameter
        mapping_function_idx = f"5\n{self.length_scale}"
        self._run_multiwfn(mapping_function_idx=mapping_function_idx, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceElectronDelocalizationRangeFunctionMean(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature
    "surface_electron_delocalization_range_function_mean", calculated with multiwfn.

    The index of this feature is 372 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    length_scale: float

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_electron_delocalization_range_function_mean``
        feature."""
        id_string = "ElectronDelocalizationRangeFunction"
        feature_name = "electron_delocalization_range_function"

        # Add additional option to mapping_function_idx parameter
        mapping_function_idx = f"5\n{self.length_scale}"
        self._run_multiwfn(mapping_function_idx=mapping_function_idx, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceElectronDelocalizationRangeFunctionMin(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature
    "surface_electron_delocalization_range_function_min", calculated with multiwfn.

    The index of this feature is 373 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    length_scale: float

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_electron_delocalization_range_function_min``
        feature."""
        id_string = "ElectronDelocalizationRangeFunction"
        feature_name = "electron_delocalization_range_function"

        # Add additional option to mapping_function_idx parameter
        mapping_function_idx = f"5\n{self.length_scale}"
        self._run_multiwfn(mapping_function_idx=mapping_function_idx, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceElectronDensityMax(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_electron_density_max", calculated with
    multiwfn.

    The index of this feature is 374 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_electron_density_max`` feature."""
        id_string = "ElectronDensity"
        feature_name = "electron_density"
        self._run_multiwfn(mapping_function_idx=11, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceElectronDensityMean(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_electron_density_mean", calculated with
    multiwfn.

    The index of this feature is 375 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_electron_density_mean`` feature."""
        id_string = "ElectronDensity"
        feature_name = "electron_density"
        self._run_multiwfn(mapping_function_idx=11, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceElectronDensityMin(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_electron_density_min", calculated with
    multiwfn.

    The index of this feature is 376 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_electron_density_min`` feature."""
        id_string = "ElectronDensity"
        feature_name = "electron_density"
        self._run_multiwfn(mapping_function_idx=11, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceElectrostaticPotentialBalanceOfCharges(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature
    "surface_electrostatic_potential_balance_of_charges", calculated with multiwfn.

    The index of this feature is 377 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_electrostatic_potential_balance_of_charges``
        feature."""
        id_string = "ElectrostaticPotential"
        feature_name = "electrostatic_potential"
        self._run_multiwfn(mapping_function_idx=1, id_string=id_string)
        self._read_output_file3(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceElectrostaticPotentialBalanceOfChargesTimesVariance(
    _Multiwfn3DAtomSurface
):
    """Feature factory for the 3D atom feature
    "surface_electrostatic_potential_balance_of_charges_times_variance", calculated with
    multiwfn.

    The index of this feature is 378 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-surface_electrostatic_potential_balance_of_charges_times_variance``
        feature."""
        id_string = "ElectrostaticPotential"
        feature_name = "electrostatic_potential"
        self._run_multiwfn(mapping_function_idx=1, id_string=id_string)
        self._read_output_file3(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceElectrostaticPotentialInternalChargeSeparation(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature
    "surface_electrostatic_potential_internal_charge_separation", calculated with multiwfn.

    The index of this feature is 379 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-surface_electrostatic_potential_internal_charge_separation``
        feature."""
        id_string = "ElectrostaticPotential"
        feature_name = "electrostatic_potential"
        self._run_multiwfn(mapping_function_idx=1, id_string=id_string)
        self._read_output_file3(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceElectrostaticPotentialMax(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_electrostatic_potential_max", calculated
    with multiwfn.

    The index of this feature is 380 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_electrostatic_potential_max`` feature."""
        id_string = "ElectrostaticPotential"
        feature_name = "electrostatic_potential"
        self._run_multiwfn(mapping_function_idx=1, id_string=id_string)
        self._read_output_file3(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceElectrostaticPotentialMean(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_electrostatic_potential_mean",
    calculated with multiwfn.

    The index of this feature is 381 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_electrostatic_potential_mean`` feature."""
        id_string = "ElectrostaticPotential"
        feature_name = "electrostatic_potential"
        self._run_multiwfn(mapping_function_idx=1, id_string=id_string)
        self._read_output_file3(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceElectrostaticPotentialMin(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_electrostatic_potential_min", calculated
    with multiwfn.

    The index of this feature is 382 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_electrostatic_potential_min`` feature."""
        id_string = "ElectrostaticPotential"
        feature_name = "electrostatic_potential"
        self._run_multiwfn(mapping_function_idx=1, id_string=id_string)
        self._read_output_file3(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceLocalElectronAffinityMax(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_local_electron_affinity_max", calculated
    with multiwfn.

    The index of this feature is 383 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_local_electron_affinity_max`` feature."""
        id_string = "LocalElectronAffinity"
        feature_name = "local_electron_affinity"
        self._run_multiwfn(mapping_function_idx=4, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceLocalElectronAffinityMean(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_local_electron_affinity_mean",
    calculated with multiwfn.

    The index of this feature is 384 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_local_electron_affinity_mean`` feature."""
        id_string = "LocalElectronAffinity"
        feature_name = "local_electron_affinity"
        self._run_multiwfn(mapping_function_idx=4, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceLocalElectronAffinityMin(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_local_electron_affinity_min", calculated
    with multiwfn.

    The index of this feature is 385 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_local_electron_affinity_min`` feature."""
        id_string = "LocalElectronAffinity"
        feature_name = "local_electron_affinity"
        self._run_multiwfn(mapping_function_idx=4, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceLocalElectronAttachmentEnergyMax(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_local_electron_attachment_energy_max",
    calculated with multiwfn.

    The index of this feature is 386 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_local_electron_attachment_energy_max``
        feature."""
        id_string = "LocalElectronAttachmentEnergy"
        feature_name = "local_electron_attachment_energy"
        self._run_multiwfn(mapping_function_idx=-4, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceLocalElectronAttachmentEnergyMean(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_local_electron_attachment_energy_mean",
    calculated with multiwfn.

    The index of this feature is 387 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_local_electron_attachment_energy_mean``
        feature."""
        id_string = "LocalElectronAttachmentEnergy"
        feature_name = "local_electron_attachment_energy"
        self._run_multiwfn(mapping_function_idx=-4, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceLocalElectronAttachmentEnergyMin(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_local_electron_attachment_energy_min",
    calculated with multiwfn.

    The index of this feature is 388 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_local_electron_attachment_energy_min``
        feature."""
        id_string = "LocalElectronAttachmentEnergy"
        feature_name = "local_electron_attachment_energy"
        self._run_multiwfn(mapping_function_idx=-4, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceOrbitalOverlapLengthFunctionMax(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_orbital_overlap_length_function_max",
    calculated with multiwfn.

    The index of this feature is 389 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    orbital_overlap_edr_option: List[Union[int, float]]

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_orbital_overlap_length_function_max``
        feature."""
        id_string = "OrbitalOverlapLengthFunction"
        feature_name = "orbital_overlap_length_function"

        # Add additional option to mapping_function_idx parameter
        mapping_function_idx = (
            f"6\n1\n{self.orbital_overlap_edr_option[0]}\n"
            f"{self.orbital_overlap_edr_option[1]}\n{self.orbital_overlap_edr_option[2]}\n"
        )

        self._run_multiwfn(mapping_function_idx=mapping_function_idx, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceOrbitalOverlapLengthFunctionMean(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_orbital_overlap_length_function_mean",
    calculated with multiwfn.

    The index of this feature is 390 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    orbital_overlap_edr_option: List[Union[int, float]]

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_orbital_overlap_length_function_mean``
        feature."""
        id_string = "OrbitalOverlapLengthFunction"
        feature_name = "orbital_overlap_length_function"

        # Add additional option to mapping_function_idx parameter
        mapping_function_idx = (
            f"6\n1\n{self.orbital_overlap_edr_option[0]}\n"
            f"{self.orbital_overlap_edr_option[1]}\n{self.orbital_overlap_edr_option[2]}\n"
        )

        self._run_multiwfn(mapping_function_idx=mapping_function_idx, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceOrbitalOverlapLengthFunctionMin(_Multiwfn3DAtomSurface):
    """Feature factory for the 3D atom feature "surface_orbital_overlap_length_function_min",
    calculated with multiwfn.

    The index of this feature is 391 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    orbital_overlap_edr_option: List[Union[int, float]]

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-surface_orbital_overlap_length_function_min``
        feature."""
        id_string = "OrbitalOverlapLengthFunction"
        feature_name = "orbital_overlap_length_function"

        # Add additional option to mapping_function_idx parameter
        mapping_function_idx = (
            f"6\n1\n{self.orbital_overlap_edr_option[0]}\n"
            f"{self.orbital_overlap_edr_option[1]}\n{self.orbital_overlap_edr_option[2]}\n"
        )

        self._run_multiwfn(mapping_function_idx=mapping_function_idx, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceSignSecondLargestEigenvalueElectronDensityHessianMax(
    _Multiwfn3DAtomSurface
):
    """Feature factory for the 3D atom feature
    "surface_sign_second_largest_eigenvalue_electron_density_hessian_max", calculated with
    multiwfn.

    The index of this feature is 392 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-surface_sign_second_largest_eigenvalue_electron_density_hessian_max``
        feature."""
        id_string = "SignSecondLargestEigenvalueElectronDensityHessian"
        feature_name = "sign_second_largest_eigenvalue_electron_density_hessian"
        self._run_multiwfn(mapping_function_idx=12, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceSignSecondLargestEigenvalueElectronDensityHessianMean(
    _Multiwfn3DAtomSurface
):
    """Feature factory for the 3D atom feature
    "surface_sign_second_largest_eigenvalue_electron_density_hessian_mean", calculated with
    multiwfn.

    The index of this feature is 393 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-surface_sign_second_largest_eigenvalue_electron_density_hessian_mean``
        feature."""
        id_string = "SignSecondLargestEigenvalueElectronDensityHessian"
        feature_name = "sign_second_largest_eigenvalue_electron_density_hessian"
        self._run_multiwfn(mapping_function_idx=12, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)


class Multiwfn3DAtomSurfaceSignSecondLargestEigenvalueElectronDensityHessianMin(
    _Multiwfn3DAtomSurface
):
    """Feature factory for the 3D atom feature
    "surface_sign_second_largest_eigenvalue_electron_density_hessian_min", calculated with
    multiwfn.

    The index of this feature is 394 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.surface" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-surface_sign_second_largest_eigenvalue_electron_density_hessian_min``
        feature."""
        id_string = "SignSecondLargestEigenvalueElectronDensityHessian"
        feature_name = "sign_second_largest_eigenvalue_electron_density_hessian"
        self._run_multiwfn(mapping_function_idx=12, id_string=id_string)
        self._read_output_file2(id_string=id_string, feature_name=feature_name)
