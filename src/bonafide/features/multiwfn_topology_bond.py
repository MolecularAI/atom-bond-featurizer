"""Bond topology features from ``Multiwfn``."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import numpy as np

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver
from bonafide.utils.multiwfn_properties import read_prop_file

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from rdkit import Chem


class _Multiwfn3DBondTopology(BaseFeaturizer):
    """Parent feature factory for the 3D bond topology Multiwfn features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    neighbor_distance_cutoff: float
    step_size: float

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the Multiwfn bond topology features."""
        self._run_multiwfn()
        self._read_cp_prop_file()

    def _run_multiwfn(self) -> None:
        """Run Multiwfn.

        Returns
        -------
        None
        """
        # Select topology analysis
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [2]

        # Add path generation options
        multiwfn_commands.extend([-2, 2, self.step_size, 3, self.neighbor_distance_cutoff, 0])

        # Search critical points
        multiwfn_commands.extend([2, 3])

        # Generate paths
        multiwfn_commands.append(8)

        # Print results to separate text file
        multiwfn_commands.extend([7, 0])

        # Exit program
        multiwfn_commands.extend([-10, "q"])

        # Set up environment variables
        environment_variables = {
            var: getattr(self, var, None) for var in PROGRAM_ENVIRONMENT_VARIABLES["multiwfn"]
        }

        # Run Multiwfn
        multiwfn_driver(
            cmds=multiwfn_commands,
            input_file_path=str(self.electronic_struc_n),
            output_file_name=f"Multiwfn3DBondTopology__{self.conformer_name}",
            environment_variables=environment_variables,
            namespace=self.conformer_name[::-1].split("__", 1)[-1][::-1],
        )

    def _read_cp_prop_file(self) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"CPprop__{self.conformer_name}.txt"
        if os.path.isfile("CPprop.txt") is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Rename CPprop.txt file
        os.rename("CPprop.txt", f"CPprop__{self.conformer_name}.txt")

        # Open output file and read data
        with open(_opath, "r") as f:
            cp_prop_output = f.readlines()

        all_data = read_prop_file(file_content=cp_prop_output, prefix="bcp_")

        for data in all_data:
            assert isinstance(data["_atoms"], tuple)  # for type checker
            atom_idx_1 = data["_atoms"][0] - 1
            atom_idx_2 = data["_atoms"][1] - 1

            # Find the bond in the mol object and save the data
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
                    # Add data that needs to be separately calculated
                    distance_start_atom, start_atom_coordinates = self._get_distance(
                        data=data, to="start", bond_obj=bond, mol_obj=self.mol
                    )
                    distance_end_atom, end_atom_coordinates = self._get_distance(
                        data=data, to="end", bond_obj=bond, mol_obj=self.mol
                    )
                    relative_bond_position = self._get_relative_bond_position(
                        start_atom_coordinates=start_atom_coordinates,
                        end_atom_coordinates=end_atom_coordinates,
                        distance_start_atom=distance_start_atom,
                    )
                    data["bcp_distance_start_atom"] = distance_start_atom
                    data["bcp_distance_end_atom"] = distance_end_atom
                    data["bcp_relative_bond_position"] = relative_bond_position

                    # Correct data types are ensured -> typing is ignored here
                    self.results[bond_idx] = {
                        f"multiwfn3D-bond-topology_{feature_id_str}": value  # type: ignore[misc]
                        for feature_id_str, value in data.items()
                        if feature_id_str not in ["_atoms", "_not_found"]
                    }
                    break

    @staticmethod
    def _get_distance(
        data: Dict[str, Any], to: str, bond_obj: Chem.rdchem.Bond, mol_obj: Chem.rdchem.Mol
    ) -> Tuple[float, NDArray[np.float64]]:
        """Calculate the distance of a bond critical point to one of the atoms forming the bond.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary containing the coordinates of the bond critical point.
        to : str
            Whether to calculate the distance to the "start" or "end" atom of the bond.
        bond_obj : Chem.rdchem.Bond
            The RDKit bond object.
        mol_obj : Chem.rdchem.Mol
            The RDKit molecule object.

        Returns
        -------
        Tuple[float, NDArray[np.float64]]
            The distance of the bond critical point to the respective atom and that atom's
            cartesian coordinates.
        """
        bcp_coordinates = np.array([float(x) for x in data["bcp_coordinates"].split(",")])

        if to == "start":
            atom_idx = bond_obj.GetBeginAtomIdx()
        if to == "end":
            atom_idx = bond_obj.GetEndAtomIdx()

        pos = mol_obj.GetConformer().GetAtomPosition(atom_idx)
        atom_coordinates = np.array([pos.x, pos.y, pos.z])

        distance = np.linalg.norm(bcp_coordinates - atom_coordinates)
        return float(distance), atom_coordinates

    @staticmethod
    def _get_relative_bond_position(
        start_atom_coordinates: NDArray[np.float64],
        end_atom_coordinates: NDArray[np.float64],
        distance_start_atom: float,
    ) -> float:
        """Calculate the relative position of a bond critical point along its respective bond.

        Parameters
        ----------
        start_atom_coordinates : NDArray[np.float64]
            The cartesian coordinates of the start atom of the bond.
        end_atom_coordinates : NDArray[np.float64]
            The cartesian coordinates of the end atom of the bond.
        distance_start_atom : float
            The distance of the bond critical point to the start atom of the bond.

        Returns
        -------
        float
            The ratio between the bond length and the distance of the bond critical point to
            the start atom of the bond.
        """
        bond_length = np.linalg.norm(end_atom_coordinates - start_atom_coordinates)
        ratio = distance_start_atom / bond_length
        return float(ratio)


class Multiwfn3DBondTopologyBcpAverageLocalIonizationEnergy(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_average_local_ionization_energy",
    calculated with multiwfn.

    The index of this feature is 446 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpCoordinates(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_coordinates", calculated with
    multiwfn.

    The index of this feature is 447 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpDeltaGHirshfeld(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_delta_g_hirshfeld", calculated with
    multiwfn.

    The index of this feature is 448 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpDeltaGPromolecular(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_delta_g_promolecular", calculated
    with multiwfn.

    The index of this feature is 449 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpDistanceEndAtom(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_distance_end_atom", calculated with
    multiwfn.

    The index of this feature is 450 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpDistanceStartAtom(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_distance_start_atom", calculated
    with multiwfn.

    The index of this feature is 451 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpElectronDensity(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_electron_density", calculated with
    multiwfn.

    The index of this feature is 452 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpElectronDensityAlpha(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_electron_density_alpha", calculated
    with multiwfn.

    The index of this feature is 453 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpElectronDensityBeta(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_electron_density_beta", calculated
    with multiwfn.

    The index of this feature is 454 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpElectronDensityEllipticity(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_electron_density_ellipticity",
    calculated with multiwfn.

    The index of this feature is 455 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpElectronLocalizationFunction(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_electron_localization_function",
    calculated with multiwfn.

    The index of this feature is 456 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpElectrostaticPotential(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_electrostatic_potential",
    calculated with multiwfn.

    The index of this feature is 457 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpElectrostaticPotentialElectrons(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_electrostatic_potential_electrons",
    calculated with multiwfn.

    The index of this feature is 458 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpElectrostaticPotentialNuclearCharges(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature
    "topology_bcp_electrostatic_potential_nuclear_charges", calculated with multiwfn.

    The index of this feature is 459 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpEnergyDensity(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_energy_density", calculated with
    multiwfn.

    The index of this feature is 460 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpEtaIndex(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_eta_index", calculated with
    multiwfn.

    The index of this feature is 461 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpGradientComponentsXYZ(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_gradient_components_x_y_z",
    calculated with multiwfn.

    The index of this feature is 462 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpGradientNorm(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_gradient_norm", calculated with
    multiwfn.

    The index of this feature is 463 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpHamiltonianKineticEnergy(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_hamiltonian_kinetic_energy",
    calculated with multiwfn.

    The index of this feature is 464 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpHessianDeterminant(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_hessian_determinant", calculated
    with multiwfn.

    The index of this feature is 465 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()


class Multiwfn3DBondTopologyBcpHessianEigenvalues(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_hessian_eigenvalues", calculated
    with multiwfn.

    The index of this feature is 466 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpInteractionRegionIndicator(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_interaction_region_indicator",
    calculated with multiwfn.

    The index of this feature is 467 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpLagrangianKineticEnergy(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_lagrangian_kinetic_energy",
    calculated with multiwfn.

    The index of this feature is 468 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpLaplacianComponentsXYZ(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_laplacian_components_x_y_z",
    calculated with multiwfn.

    The index of this feature is 469 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpLaplacianOfElectronDensity(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_laplacian_of_electron_density",
    calculated with multiwfn.

    The index of this feature is 470 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpLaplacianTotal(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_laplacian_total", calculated with
    multiwfn.

    The index of this feature is 471 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpLocalInformationEntropy(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_local_information_entropy",
    calculated with multiwfn.

    The index of this feature is 472 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpLocalizedOrbitalLocator(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_localized_orbital_locator",
    calculated with multiwfn.

    The index of this feature is 473 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpPotentialEnergyDensity(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_potential_energy_density",
    calculated with multiwfn.

    The index of this feature is 474 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpReducedDensityGradient(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_reduced_density_gradient",
    calculated with multiwfn.

    The index of this feature is 475 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpReducedDensityGradientPromolecular(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature
    "topology_bcp_reduced_density_gradient_promolecular", calculated with multiwfn.

    The index of this feature is 476 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpRelativeBondPosition(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_relative_bond_position", calculated
    with multiwfn.

    The index of this feature is 477 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpSignSecondLargestEigenvalueElectronDensityHessian(
    _Multiwfn3DBondTopology
):
    """Feature factory for the 3D bond feature
    "topology_bcp_sign_second_largest_eigenvalue_electron_density_hessian", calculated with
    multiwfn.

    The index of this feature is 478 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpSignSecondLargestEigenvalueElectronDensityHessianPromolecular(
    _Multiwfn3DBondTopology
):
    """Feature factory for the 3D bond feature
    "topology_bcp_sign_second_largest_eigenvalue_electron_density_hessian_promolecular",
    calculated with multiwfn.

    The index of this feature is 479 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpSpinDensity(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_spin_density", calculated with
    multiwfn.

    The index of this feature is 480 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyBcpVdwPotential(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_bcp_vdw_potential", calculated with
    multiwfn.

    The index of this feature is 481 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyCpIndex(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_cp_index", calculated with multiwfn.

    The index of this feature is 482 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology


class Multiwfn3DBondTopologyCpType(_Multiwfn3DBondTopology):
    """Feature factory for the 3D bond feature "topology_cp_type", calculated with multiwfn.

    The index of this feature is 483 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.topology" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Multiwfn3DBondTopology
