"""Features from ``DScribe``."""

from typing import Any, List, Union

import numpy as np
from ase.io import read
from dscribe.descriptors import ACSF, LMBTR, SOAP, CoulombMatrix
from rdkit import Chem

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.io_ import write_xyz_file_from_coordinates_array


class _Dscribe3DAtom(BaseFeaturizer):
    """Parent feature factory for the 3D atom DScribe features."""

    # Common attributes for child classes
    species: Union[str, List[str]]
    r_cut: float

    # ACSF parameters
    g2_params: Any
    g3_params: Any
    g4_params: Any
    g5_params: Any

    # Coulomb matrix parameters
    scaling_exponent: float

    # LMBTR parameters
    geometry_function: str
    grid_min: float
    grid_max: float
    grid_sigma: float
    grid_n: float
    normalize_gaussians: bool
    normalization: bool
    weighting_function: bool
    weighting_scale: float
    weighting_threshold: float

    # SOAP parameters
    average: str
    l_max: int
    n_max: int
    rbf: str
    sigma: float

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def _write_input_file(self) -> None:
        """Write an XYZ file as input for DScribe.

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


class Dscribe3DAtomAtomCenteredSymmetryFunction(_Dscribe3DAtom):
    """Feature factory for the 3D atom feature "atom_centered_symmetry_function", calculated
    with dscribe.

    The index of this feature is 93 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "dscribe.acsf" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``dscribe3D-atom-atom_centered_symmetry_function`` feature."""
        # Write input file for ase/DScribe
        self._write_input_file()

        # Read in molecule
        ase_mol = read(filename=f"{self.conformer_name}.xyz")
        if isinstance(ase_mol, list):
            ase_mol = ase_mol[0]

        # Define chemical symbols to include
        if self.species == "auto":
            chem_symbol_list = list(set(ase_mol.get_chemical_symbols()))  # type: ignore[no-untyped-call]
        else:
            chem_symbol_list = self.species  # type: ignore[assignment]

        # Calculate features and write them to the results dictionary
        acsf = ACSF(
            r_cut=self.r_cut,
            g2_params=self.g2_params,
            g3_params=self.g3_params,
            g4_params=self.g4_params,
            g5_params=self.g5_params,
            species=chem_symbol_list,
        )
        feature_matrix = acsf.create(ase_mol)

        for idx, row in enumerate(feature_matrix):
            vec = ",".join([str(float(x)) for x in row])
            self.results[idx] = {self.feature_name: vec}


class Dscribe3DAtomAtomicCoulombVector(_Dscribe3DAtom):
    """Feature factory for the 3D atom feature "atomic_coulomb_vector", calculated with dscribe.

    The index of this feature is 94 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "dscribe.coulomb_matrix" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``dscribe3D-atom-atomic_coulomb_vector`` feature."""
        # Write input file for ase/DScribe
        self._write_input_file()

        # Read in molecule
        ase_mol = read(f"{self.conformer_name}.xyz")
        if isinstance(ase_mol, list):
            ase_mol = ase_mol[0]

        n_atoms = len(ase_mol.get_chemical_symbols())  # type: ignore[no-untyped-call]

        # Get the Coulomb matrix, either from the cache or by calculating it
        _feature_name = "dscribe3d-global-coulomb_matrix"
        if _feature_name not in self.global_feature_cache[self.conformer_idx]:
            cm = CoulombMatrix(n_atoms_max=n_atoms, permutation="none")
            coulomb_matrix = cm.create(ase_mol)
            coulomb_matrix = coulomb_matrix.reshape(n_atoms, n_atoms)
            self.global_feature_cache[self.conformer_idx][_feature_name] = coulomb_matrix
        else:
            coulomb_matrix = self.global_feature_cache[self.conformer_idx][_feature_name]

        # Get the distance matrix, either from the cache or by calculating it
        _feature_name = "rdkit3d-global-distance_matrix"
        if _feature_name not in self.global_feature_cache[self.conformer_idx]:
            distance_matrix = Chem.Get3DDistanceMatrix(self.mol)
            self.global_feature_cache[self.conformer_idx][_feature_name] = distance_matrix
        else:
            distance_matrix = self.global_feature_cache[self.conformer_idx][_feature_name]

        # Calculate vectors and write them to the results dictionary
        for idx, coulomb_vector in enumerate(coulomb_matrix):
            dist_vector = distance_matrix[idx]
            coulomb_vector = coulomb_vector[np.argsort(a=dist_vector)]

            scale_vector = 1 / (np.sort(a=dist_vector)[1:] ** self.scaling_exponent)
            scale_vector = np.insert(arr=scale_vector, obj=0, values=1.0)

            values = coulomb_vector * scale_vector
            vec = ",".join([str(float(x)) for x in values])
            self.results[idx] = {self.feature_name: vec}


class Dscribe3DAtomLocalManyBodyTensorRepresentation(_Dscribe3DAtom):
    """Feature factory for the 3D atom feature "local_many_body_tensor_representation",
    calculated with dscribe.

    The index of this feature is 95 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "dscribe.lmbtr" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``dscribe3D-atom-local_many_body_tensor_representation`` feature."""
        # Write input file for ase/DScribe
        self._write_input_file()

        # Read in molecule
        ase_mol = read(f"{self.conformer_name}.xyz")
        if isinstance(ase_mol, list):
            ase_mol = ase_mol[0]

        # Define chemical symbols to include
        if self.species == "auto":
            chem_symbol_list = list(set(ase_mol.get_chemical_symbols()))  # type: ignore[no-untyped-call]
        else:
            chem_symbol_list = self.species  # type: ignore[assignment]

        # Calculate features and write them to the results dictionary
        lmbtr = LMBTR(
            geometry={"function": self.geometry_function},
            grid={
                "min": self.grid_min,
                "max": self.grid_max,
                "sigma": self.grid_sigma,
                "n": self.grid_n,
            },
            weighting={
                "function": self.weighting_function,
                "scale": self.weighting_scale,
                "threshold": self.weighting_threshold,
            },
            normalize_gaussians=self.normalize_gaussians,
            normalization=self.normalization,
            species=chem_symbol_list,
        )
        feature_matrix = lmbtr.create(ase_mol)

        for idx, row in enumerate(feature_matrix):
            vec = ",".join([str(float(x)) for x in row])
            self.results[idx] = {self.feature_name: vec}


class Dscribe3DAtomSmoothOverlapOfAtomicPositions(_Dscribe3DAtom):
    """Feature factory for the 3D atom feature "smooth_overlap_of_atomic_positions", calculated
    with dscribe.

    The index of this feature is 96 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "dscribe.soap" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``dscribe3D-atom-smooth_overlap_of_atomic_positions`` feature."""
        # Write input file for ase/DScribe
        self._write_input_file()

        # Read in molecule
        ase_mol = read(f"{self.conformer_name}.xyz")
        if isinstance(ase_mol, list):
            ase_mol = ase_mol[0]

        # Define chemical symbols to include
        if self.species == "auto":
            chem_symbol_list = list(set(ase_mol.get_chemical_symbols()))  # type: ignore[no-untyped-call]
        else:
            chem_symbol_list = self.species  # type: ignore[assignment]

        # Calculate features and write them to the results dictionary
        soap = SOAP(
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
            sigma=self.sigma,
            rbf=self.rbf,
            average=self.average,
            species=chem_symbol_list,
        )
        feature_matrix = soap.create(ase_mol)

        for idx, row in enumerate(feature_matrix):
            vec = ",".join([str(float(x)) for x in row])
            self.results[idx] = {self.feature_name: vec}
