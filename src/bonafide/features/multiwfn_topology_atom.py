"""Atom topology features from ``Multiwfn``."""

import os
from typing import List, Union

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver
from bonafide.utils.multiwfn_properties import read_prop_file


class _Multiwfn3DAtomTopology(BaseFeaturizer):
    """Parent feature factory for the 3D atom topology Multiwfn features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the Multiwfn atom topology features."""
        self._run_multiwfn()
        self._read_atom_prop_file()

    def _run_multiwfn(self) -> None:
        """Run Multiwfn.

        Returns
        -------
        None
        """
        # Select atom properties
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [1]

        # Specify atom index
        multiwfn_commands.append(f"a{self.atom_bond_idx + 1}")  # Multiwfn uses 1-based indexing

        # Exit program
        multiwfn_commands.extend(["q", "q"])

        # Set up environment variables
        environment_variables = {
            var: getattr(self, var, None) for var in PROGRAM_ENVIRONMENT_VARIABLES["multiwfn"]
        }

        # Run Multiwfn
        multiwfn_driver(
            cmds=multiwfn_commands,
            input_file_path=str(self.electronic_struc_n),
            output_file_name=f"Multiwfn3DAtomTopology__{self.conformer_name}__"
            f"atom-{self.atom_bond_idx}",
            environment_variables=environment_variables,
            namespace=self.conformer_name[::-1].split("__", 1)[-1][::-1],
        )

    def _read_atom_prop_file(self) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"Multiwfn3DAtomTopology__{self.conformer_name}__atom-{self.atom_bond_idx}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file and read data
        with open(_opath, "r") as f:
            prop_output = f.readlines()

        all_data = read_prop_file(file_content=prop_output)

        if len(all_data) != 1:
            self._err = (
                "the data extraction from the Multiwfn output file was not successful. "
                "Check your input and the produced output files"
            )
            return

        data = all_data[0]

        # Correct data types are ensured -> typing is ignored here
        self.results[self.atom_bond_idx] = {
            f"multiwfn3D-atom-topology_{feature_id_str}": value  # type: ignore[misc]
            for feature_id_str, value in data.items()
            if feature_id_str not in ["_not_found"]
        }


class Multiwfn3DAtomTopologyAverageLocalIonizationEnergy(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_average_local_ionization_energy",
    calculated with multiwfn.

    The index of this feature is 395 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyDeltaGHirshfeld(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_delta_g_hirshfeld", calculated with
    multiwfn.

    The index of this feature is 396 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyDeltaGPromolecular(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_delta_g_promolecular", calculated with
    multiwfn.

    The index of this feature is 397 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyElectronDensity(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_electron_density", calculated with
    multiwfn.

    The index of this feature is 398 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyElectronDensityAlpha(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_electron_density_alpha", calculated
    with multiwfn.

    The index of this feature is 399 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyElectronDensityBeta(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_electron_density_beta", calculated with
    multiwfn.

    The index of this feature is 400 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyElectronDensityEllipticity(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_electron_density_ellipticity",
    calculated with multiwfn.

    The index of this feature is 401 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyElectronLocalizationFunction(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_electron_localization_function",
    calculated with multiwfn.

    The index of this feature is 402 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyElectrostaticPotential(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_electrostatic_potential", calculated
    with multiwfn.

    The index of this feature is 403 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyElectrostaticPotentialElectrons(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_electrostatic_potential_electrons",
    calculated with multiwfn.

    The index of this feature is 404 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyElectrostaticPotentialNuclearCharges(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature
    "topology_electrostatic_potential_nuclear_charges", calculated with multiwfn.

    The index of this feature is 405 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyEnergyDensity(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_energy_density", calculated with
    multiwfn.

    The index of this feature is 406 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyEtaIndex(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_eta_index", calculated with multiwfn.

    The index of this feature is 407 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyGradientComponentsXYZ(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_gradient_components_x_y_z", calculated
    with multiwfn.

    The index of this feature is 408 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyGradientNorm(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_gradient_norm", calculated with
    multiwfn.

    The index of this feature is 409 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyHamiltonianKineticEnergy(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_hamiltonian_kinetic_energy", calculated
    with multiwfn.

    The index of this feature is 410 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyHessianDeterminant(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_hessian_determinant", calculated with
    multiwfn.

    The index of this feature is 411 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyHessianEigenvalues(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_hessian_eigenvalues", calculated with
    multiwfn.

    The index of this feature is 412 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyInteractionRegionIndicator(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_interaction_region_indicator",
    calculated with multiwfn.

    The index of this feature is 413 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyLagrangianKineticEnergy(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_lagrangian_kinetic_energy", calculated
    with multiwfn.

    The index of this feature is 414 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyLaplacianComponentsXYZ(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_laplacian_components_x_y_z", calculated
    with multiwfn.

    The index of this feature is 415 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyLaplacianOfElectronDensity(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_laplacian_of_electron_density",
    calculated with multiwfn.

    The index of this feature is 416 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyLaplacianTotal(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_laplacian_total", calculated with
    multiwfn.

    The index of this feature is 417 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyLocalInformationEntropy(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_local_information_entropy", calculated
    with multiwfn.

    The index of this feature is 418 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyLocalizedOrbitalLocator(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_localized_orbital_locator", calculated
    with multiwfn.

    The index of this feature is 419 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyPotentialEnergyDensity(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_potential_energy_density", calculated
    with multiwfn.

    The index of this feature is 420 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyReducedDensityGradient(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_reduced_density_gradient", calculated
    with multiwfn.

    The index of this feature is 421 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyReducedDensityGradientPromolecular(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_reduced_density_gradient_promolecular",
    calculated with multiwfn.

    The index of this feature is 422 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologySignSecondLargestEigenvalueElectronDensityHessian(
    _Multiwfn3DAtomTopology
):
    """Feature factory for the 3D atom feature
    "topology_sign_second_largest_eigenvalue_electron_density_hessian", calculated with
    multiwfn.

    The index of this feature is 423 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologySignSecondLargestEigenvalueElectronDensityHessianPromolecular(
    _Multiwfn3DAtomTopology
):
    """Feature factory for the 3D atom feature
    "topology_sign_second_largest_eigenvalue_electron_density_hessian_promolecular", calculated
    with multiwfn.

    The index of this feature is 424 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologySpinDensity(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_spin_density", calculated with
    multiwfn.

    The index of this feature is 425 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology


class Multiwfn3DAtomTopologyVdwPotential(_Multiwfn3DAtomTopology):
    """Feature factory for the 3D atom feature "topology_vdw_potential", calculated with
    multiwfn.

    The index of this feature is 426 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DAtomTopology
