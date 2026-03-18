"""Bond distance features for atoms and bonds in 2D molecules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
from rdkit import Chem

from bonafide.utils.base_featurizer import BaseFeaturizer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _DistanceMatrixMixin:
    """Mixin class to provide methods for calculating and caching distance matrices for 2D and
    3D molecules.
    """

    conformer_idx: int
    global_feature_cache: List[Dict[str, Optional[Union[str, bool, int, float]]]]
    mol: Chem.rdchem.Mol

    def _get_distance_matrix_2D(self) -> NDArray[np.float64]:
        """Get the 2D bond distance matrix for the molecule, either from the cache or by
        calculating it.

        Returns
        -------
        NDArray[np.float64]
            The 2D bond distance matrix of the molecule.
        """
        _feature_name = "rdkit2d-global-bond_distance_matrix"
        if _feature_name not in self.global_feature_cache[self.conformer_idx]:
            distance_matrix = Chem.GetDistanceMatrix(self.mol)
            self.global_feature_cache[self.conformer_idx][_feature_name] = distance_matrix
        else:
            distance_matrix = self.global_feature_cache[self.conformer_idx][_feature_name]

        return np.asarray(a=distance_matrix, dtype=np.float64)

    def _get_distance_matrix_3D(self) -> NDArray[np.float64]:
        """Get the 3D distance matrix for the molecule, either from the cache or by calculating it.

        Returns
        -------
        NDArray[np.float64]
            The 3D distance matrix of the molecule.
        """
        _feature_name = "rdkit3d-global-distance_matrix"
        if _feature_name not in self.global_feature_cache[self.conformer_idx]:
            distance_matrix = Chem.Get3DDistanceMatrix(self.mol)
            self.global_feature_cache[self.conformer_idx][_feature_name] = distance_matrix
        else:
            distance_matrix = self.global_feature_cache[self.conformer_idx][_feature_name]

        return np.asarray(a=distance_matrix, dtype=np.float64)


class _BonafideAtomDistance(_DistanceMatrixMixin, BaseFeaturizer):
    """Parent feature factory for the 2D and 3D atom distance features."""

    n_bonds_cutoff: int
    radius_cutoff: float

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _run_calculation(self, dimensionality: str) -> None:
        """Calculate the distance-based features for a specific atom, either for a 2D molecule
        (bond distance matrix) or a 3D molecule (spatial distance matrix).

        Parameters
        ----------
        dimensionality : str
            The dimensionality of the molecule ("2D" or "3D").

        Returns
        -------
        None
        """
        # Get the distance vector for the specified atom
        _cutoff: Union[int, float]
        if dimensionality == "2D":
            dist_vector = self._get_distance_matrix_2D()[self.atom_bond_idx]
            _cutoff = self.n_bonds_cutoff
            _suffix = "bonds"
        else:
            dist_vector = self._get_distance_matrix_3D()[self.atom_bond_idx]
            _cutoff = self.radius_cutoff
            _suffix = "angstrom"

        # Change the feature name to include the distance cutoff value
        self.feature_name = self.feature_name.replace("_x_", f"_{_cutoff}_")

        # Classify atoms (within or beyond the specified distance)
        atom_within_indices = [int(idx) for idx in np.where(dist_vector <= _cutoff)[0]]
        atom_beyond_indices = [int(idx) for idx in np.where(dist_vector > _cutoff)[0]]
        atom_within_indices.sort()
        atom_beyond_indices.sort()

        # Classify atoms exactly at the specified distance (for 2D only)
        if dimensionality == "2D":
            atom_exact_indices = [int(idx) for idx in np.where(dist_vector == _cutoff)[0]]
            atom_exact_indices.sort()

        # Classify bonds (within or beyond the specified distance) and calculate min distance
        # to bond
        bond_within_indices = []
        bond_beyond_indices = []
        dist_vector_bonds = []

        for bond in self.mol.GetBonds():
            begin_dist = dist_vector[bond.GetBeginAtomIdx()]
            end_dist = dist_vector[bond.GetEndAtomIdx()]
            if begin_dist <= _cutoff and end_dist <= _cutoff:
                bond_within_indices.append(bond.GetIdx())
            else:
                bond_beyond_indices.append(bond.GetIdx())

            dist_vector_bonds.append(min(begin_dist, end_dist))

        bond_within_indices.sort()
        bond_beyond_indices.sort()
        dist_vector_bonds_arr = np.array(dist_vector_bonds)

        # Write the data to the results dictionary
        self.results[self.atom_bond_idx] = {
            # Max and average distances to all atoms
            f"bonafide{dimensionality}-atom-distance_to_all_atoms_mean": float(
                round(number=np.mean(a=dist_vector), ndigits=4)
            ),
            f"bonafide{dimensionality}-atom-distance_to_all_atoms_max": float(
                round(number=np.max(a=dist_vector), ndigits=4)
            )
            if dimensionality == "3D"
            else int(np.max(a=dist_vector)),
            # Within/beyond cutoff classifications atoms (index list)
            f"bonafide{dimensionality}-atom-atoms_within_{_cutoff}_{_suffix}": ",".join(
                map(str, atom_within_indices)
            ),
            f"bonafide{dimensionality}-atom-atoms_beyond_{_cutoff}_{_suffix}": ",".join(
                map(str, atom_beyond_indices)
            ),
            # Within/beyond cutoff classifications atoms (count)
            f"bonafide{dimensionality}-atom-n_atoms_within_{_cutoff}_{_suffix}": len(
                atom_within_indices
            ),
            f"bonafide{dimensionality}-atom-n_atoms_beyond_{_cutoff}_{_suffix}": len(
                atom_beyond_indices
            ),
            # Within/beyond cutoff classifications atoms (fraction)
            f"bonafide{dimensionality}-atom-fraction_atoms_within_{_cutoff}_{_suffix}": round(
                number=len(atom_within_indices) / len(dist_vector), ndigits=4
            ),
            f"bonafide{dimensionality}-atom-fraction_atoms_beyond_{_cutoff}_{_suffix}": round(
                number=len(atom_beyond_indices) / len(dist_vector), ndigits=4
            ),
        }

        # Add data that depends on the presence of bonds
        _n_bonds = self.mol.GetNumBonds()
        if _n_bonds > 0:
            # Max and average distances to all bonds
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-atom-distance_to_all_bonds_mean"
            ] = float(round(number=np.mean(a=dist_vector_bonds_arr), ndigits=4))
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-atom-distance_to_all_bonds_max"
            ] = (
                float(round(number=np.max(a=dist_vector_bonds_arr), ndigits=4))
                if dimensionality == "3D"
                else int(np.max(a=dist_vector_bonds_arr))
            )

            # Within/beyond cutoff classifications bonds (index list)
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-atom-bonds_within_{_cutoff}_{_suffix}"
            ] = ",".join(map(str, bond_within_indices))
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-atom-bonds_beyond_{_cutoff}_{_suffix}"
            ] = ",".join(map(str, bond_beyond_indices))

            # Within/beyond cutoff classifications bonds (count)
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-atom-n_bonds_within_{_cutoff}_{_suffix}"
            ] = len(bond_within_indices)
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-atom-n_bonds_beyond_{_cutoff}_{_suffix}"
            ] = len(bond_beyond_indices)

            # Within/beyond cutoff classifications bonds (fraction)
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-atom-fraction_bonds_within_{_cutoff}_{_suffix}"
            ] = round(number=len(bond_within_indices) / self.mol.GetNumBonds(), ndigits=4)
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-atom-fraction_bonds_beyond_{_cutoff}_{_suffix}"
            ] = round(number=len(bond_beyond_indices) / self.mol.GetNumBonds(), ndigits=4)

        # Add data that is only defined for 2D molecules
        if dimensionality == "2D":
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-atom-atoms_{_cutoff}_{_suffix}_away"
            ] = ",".join(map(str, atom_exact_indices))
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-atom-n_atoms_{_cutoff}_{_suffix}_away"
            ] = len(atom_exact_indices)
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-atom-fraction_atoms_{_cutoff}_{_suffix}_away"
            ] = round(number=len(atom_exact_indices) / len(dist_vector), ndigits=4)


class Bonafide2DAtomAtomsBeyondXBonds(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "atoms_beyond_x_bonds", implemented within this
    package.

    The index of this feature is 2 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-atoms_beyond_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomAtomsWithinXBonds(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "atoms_within_x_bonds", implemented within this
    package.

    The index of this feature is 3 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-atoms_within_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomAtomsXBondsAway(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "atoms_x_bonds_away", implemented within this
    package.

    The index of this feature is 4 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-atoms_x_bonds_away`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomBondsBeyondXBonds(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "bonds_beyond_x_bonds", implemented within this
    package.

    The index of this feature is 15 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-bonds_beyond_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomBondsWithinXBonds(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "bonds_within_x_bonds", implemented within this
    package.

    The index of this feature is 16 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-bonds_within_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomDistanceToAllAtomsMax(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "distance_to_all_atoms_max", implemented within
    this package.

    The index of this feature is 18 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-distance_to_all_atoms_max`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomDistanceToAllAtomsMean(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "distance_to_all_atoms_mean", implemented within
    this package.

    The index of this feature is 19 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-distance_to_all_atoms_mean`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomDistanceToAllBondsMax(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "distance_to_all_bonds_max", implemented within
    this package.

    The index of this feature is 20 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-distance_to_all_bonds_max`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomDistanceToAllBondsMean(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "distance_to_all_bonds_mean", implemented within
    this package.

    The index of this feature is 21 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-distance_to_all_bonds_mean`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomFractionAtomsBeyondXBonds(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "fraction_atoms_beyond_x_bonds", implemented
    within this package.

    The index of this feature is 22 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-fraction_atoms_beyond_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomFractionAtomsWithinXBonds(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "fraction_atoms_within_x_bonds", implemented
    within this package.

    The index of this feature is 23 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-fraction_atoms_within_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomFractionAtomsXBondsAway(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "fraction_atoms_x_bonds_away", implemented within
    this package.

    The index of this feature is 24 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-fraction_atoms_x_bonds_away`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomFractionBondsBeyondXBonds(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "fraction_bonds_beyond_x_bonds", implemented
    within this package.

    The index of this feature is 25 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-fraction_bonds_beyond_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomFractionBondsWithinXBonds(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "fraction_bonds_within_x_bonds", implemented
    within this package.

    The index of this feature is 26 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-fraction_bonds_within_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomNAtomsBeyondXBonds(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "n_atoms_beyond_x_bonds", implemented within this
    package.

    The index of this feature is 31 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-n_atoms_beyond_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomNAtomsWithinXBonds(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "n_atoms_within_x_bonds", implemented within this
    package.

    The index of this feature is 32 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-n_atoms_within_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomNAtomsXBondsAway(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "n_atoms_x_bonds_away", implemented within this
    package.

    The index of this feature is 33 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-n_atoms_x_bonds_away`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomNBondsBeyondXBonds(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "n_bonds_beyond_x_bonds", implemented within this
    package.

    The index of this feature is 34 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-n_bonds_beyond_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DAtomNBondsWithinXBonds(_BonafideAtomDistance):
    """Feature factory for the 2D atom feature "n_bonds_within_x_bonds", implemented within this
    package.

    The index of this feature is 35 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-n_bonds_within_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide3DAtomAtomsBeyondXAngstrom(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "atoms_beyond_x_angstrom", implemented within
    this package.

    The index of this feature is 58 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-atoms_beyond_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomAtomsWithinXAngstrom(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "atoms_within_x_angstrom", implemented within
    this package.

    The index of this feature is 59 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-atoms_within_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomBondsBeyondXAngstrom(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "bonds_beyond_x_angstrom", implemented within
    this package.

    The index of this feature is 60 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-bonds_beyond_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomBondsWithinXAngstrom(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "bonds_within_x_angstrom", implemented within
    this package.

    The index of this feature is 61 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-bonds_within_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomDistanceToAllAtomsMax(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "distance_to_all_atoms_max", implemented within
    this package.

    The index of this feature is 62 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-distance_to_all_atoms_max`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomDistanceToAllAtomsMean(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "distance_to_all_atoms_mean", implemented within
    this package.

    The index of this feature is 63 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-distance_to_all_atoms_mean`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomDistanceToAllBondsMax(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "distance_to_all_bonds_max", implemented within
    this package.

    The index of this feature is 64 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-distance_to_all_bonds_max`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomDistanceToAllBondsMean(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "distance_to_all_bonds_mean", implemented within
    this package.

    The index of this feature is 65 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-distance_to_all_bonds_mean`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomFractionAtomsBeyondXAngstrom(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "fraction_atoms_beyond_x_angstrom", implemented
    within this package.

    The index of this feature is 66 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-fraction_atoms_beyond_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomFractionAtomsWithinXAngstrom(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "fraction_atoms_within_x_angstrom", implemented
    within this package.

    The index of this feature is 67 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-fraction_atoms_within_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomFractionBondsBeyondXAngstrom(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "fraction_bonds_beyond_x_angstrom", implemented
    within this package.

    The index of this feature is 68 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-fraction_bonds_beyond_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomFractionBondsWithinXAngstrom(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "fraction_bonds_within_x_angstrom", implemented
    within this package.

    The index of this feature is 69 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-fraction_bonds_within_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomNAtomsBeyondXAngstrom(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "n_atoms_beyond_x_angstrom", implemented within
    this package.

    The index of this feature is 70 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-n_atoms_beyond_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomNAtomsWithinXAngstrom(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "n_atoms_within_x_angstrom", implemented within
    this package.

    The index of this feature is 71 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-n_atoms_within_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomNBondsBeyondXAngstrom(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "n_bonds_beyond_x_angstrom", implemented within
    this package.

    The index of this feature is 72 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-n_bonds_beyond_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DAtomNBondsWithinXAngstrom(_BonafideAtomDistance):
    """Feature factory for the 3D atom feature "n_bonds_within_x_angstrom", implemented within
    this package.

    The index of this feature is 73 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-atom-n_bonds_within_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class _BonafideBondDistance(_DistanceMatrixMixin, BaseFeaturizer):
    """Parent feature factory for the 2D and 3D bond distance features."""

    n_bonds_cutoff: int
    radius_cutoff: float

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _run_calculation(self, dimensionality: str) -> None:
        """Calculate the distance-based features for a specific bond, either for a 2D molecule
        (bond distance matrix) or a 3D molecule (spatial distance matrix).

        Parameters
        ----------
        dimensionality : str
            The dimensionality of the molecule ("2D" or "3D").

        Returns
        -------
        None
        """
        # Get the distance matrix
        _cutoff: Union[int, float]
        if dimensionality == "2D":
            distance_matrix = self._get_distance_matrix_2D()
            _cutoff = self.n_bonds_cutoff
            _suffix = "bonds"
        else:
            distance_matrix = self._get_distance_matrix_3D()
            _cutoff = self.radius_cutoff
            _suffix = "angstrom"

        # Change the feature name to include the distance cutoff value
        self.feature_name = self.feature_name.replace("_x_", f"_{_cutoff}_")

        # Get the distance vectors for the two atoms in the bond
        _bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        begin_dist_vector = distance_matrix[_bond.GetBeginAtomIdx()]
        end_dist_vector = distance_matrix[_bond.GetEndAtomIdx()]
        min_dist_vector = np.minimum(begin_dist_vector, end_dist_vector)

        # Classify atoms (within or beyond the specified distance)
        begin_atom_within_indices = [int(idx) for idx in np.where(begin_dist_vector <= _cutoff)[0]]
        begin_atom_beyond_indices = [int(idx) for idx in np.where(begin_dist_vector > _cutoff)[0]]

        end_atom_within_indices = [int(idx) for idx in np.where(end_dist_vector <= _cutoff)[0]]
        end_atom_beyond_indices = [int(idx) for idx in np.where(end_dist_vector > _cutoff)[0]]

        # 2D case: an atom is within the cutoff if it is within the cutoff from either of the two
        # atoms in the reference bond
        if dimensionality == "2D":
            atom_within_indices = list(set(begin_atom_within_indices + end_atom_within_indices))
            atom_beyond_indices = list(
                set(begin_atom_beyond_indices).intersection(set(end_atom_beyond_indices))
            )

        # 3D case: an atom is within the cutoff only if it is within the cutoff from both of the
        # two atoms in the reference bond
        else:
            atom_within_indices = list(
                set(begin_atom_within_indices).intersection(set(end_atom_within_indices))
            )
            atom_beyond_indices = list(set(begin_atom_beyond_indices + end_atom_beyond_indices))

        atom_within_indices.sort()
        atom_beyond_indices.sort()

        # Classify atoms exactly at the specified distance (for 2D only)
        if dimensionality == "2D":
            begin_atom_exact_indices = [
                int(idx) for idx in np.where(begin_dist_vector == _cutoff)[0]
            ]
            end_atom_exact_indices = [int(idx) for idx in np.where(end_dist_vector == _cutoff)[0]]
            atom_exact_indices = list(set(begin_atom_exact_indices + end_atom_exact_indices))
            atom_exact_indices.sort()

        # Classify bonds (within or beyond the specified distance)
        begin_bond_within_indices_2D = []
        begin_bond_beyond_indices_2D = []
        end_bond_within_indices_2D = []
        end_bond_beyond_indices_2D = []

        bond_within_indices_3D = []
        bond_beyond_indices_3D = []

        bond_dist_vector = []

        for bond in self.mol.GetBonds():
            begin_begin_dist = begin_dist_vector[bond.GetBeginAtomIdx()]
            begin_end_dist = begin_dist_vector[bond.GetEndAtomIdx()]
            end_begin_dist = end_dist_vector[bond.GetBeginAtomIdx()]
            end_end_dist = end_dist_vector[bond.GetEndAtomIdx()]

            # Calculate the minimum distance from the reference bond to the current bond
            bond_dist_vector.append(
                min(begin_begin_dist, begin_end_dist, end_begin_dist, end_end_dist)
            )

            # 2D case: both atoms of the other bond must be within the cutoff distance
            # either from the begin or end atom of the reference bond
            if dimensionality == "2D":
                if begin_begin_dist <= _cutoff and begin_end_dist <= _cutoff:
                    begin_bond_within_indices_2D.append(bond.GetIdx())
                else:
                    begin_bond_beyond_indices_2D.append(bond.GetIdx())

                if end_begin_dist <= _cutoff and end_end_dist <= _cutoff:
                    end_bond_within_indices_2D.append(bond.GetIdx())
                else:
                    end_bond_beyond_indices_2D.append(bond.GetIdx())
                continue

            # 3D case: both atoms of the other bond must be within the cutoff distance
            # from both the begin and end atoms of the reference bond
            if all(
                [
                    begin_begin_dist <= _cutoff,
                    begin_end_dist <= _cutoff,
                    end_begin_dist <= _cutoff,
                    end_end_dist <= _cutoff,
                ]
            ):
                bond_within_indices_3D.append(bond.GetIdx())
            else:
                bond_beyond_indices_3D.append(bond.GetIdx())

        if dimensionality == "2D":
            bond_within_indices = list(
                set(begin_bond_within_indices_2D + end_bond_within_indices_2D)
            )
            bond_beyond_indices = list(
                set(begin_bond_beyond_indices_2D).intersection(set(end_bond_beyond_indices_2D))
            )

        else:
            bond_within_indices = bond_within_indices_3D
            bond_beyond_indices = bond_beyond_indices_3D

        bond_within_indices.sort()
        bond_beyond_indices.sort()

        # Write the data to the results dictionary
        self.results[self.atom_bond_idx] = {
            # Max and average distances to all atoms
            f"bonafide{dimensionality}-bond-distance_to_all_atoms_mean": float(
                round(number=np.mean(a=min_dist_vector), ndigits=4)
            ),
            f"bonafide{dimensionality}-bond-distance_to_all_atoms_max": float(
                round(number=np.max(a=min_dist_vector), ndigits=4)
            )
            if dimensionality == "3D"
            else int(np.max(a=min_dist_vector)),
            # Within/beyond cutoff classifications atoms (index list)
            f"bonafide{dimensionality}-bond-atoms_within_{_cutoff}_{_suffix}": ",".join(
                map(str, atom_within_indices)
            ),
            f"bonafide{dimensionality}-bond-bonds_within_{_cutoff}_{_suffix}": ",".join(
                map(str, bond_within_indices)
            ),
            f"bonafide{dimensionality}-bond-atoms_beyond_{_cutoff}_{_suffix}": ",".join(
                map(str, atom_beyond_indices)
            ),
            f"bonafide{dimensionality}-bond-bonds_beyond_{_cutoff}_{_suffix}": ",".join(
                map(str, bond_beyond_indices)
            ),
            # Within/beyond cutoff classifications atoms (count)
            f"bonafide{dimensionality}-bond-n_atoms_within_{_cutoff}_{_suffix}": len(
                atom_within_indices
            ),
            f"bonafide{dimensionality}-bond-n_bonds_within_{_cutoff}_{_suffix}": len(
                bond_within_indices
            ),
            f"bonafide{dimensionality}-bond-n_atoms_beyond_{_cutoff}_{_suffix}": len(
                atom_beyond_indices
            ),
            f"bonafide{dimensionality}-bond-n_bonds_beyond_{_cutoff}_{_suffix}": len(
                bond_beyond_indices
            ),
            # Within/beyond cutoff classifications atoms (fraction)
            f"bonafide{dimensionality}-bond-fraction_atoms_within_{_cutoff}_{_suffix}": round(
                number=len(atom_within_indices) / len(begin_dist_vector), ndigits=4
            ),
            f"bonafide{dimensionality}-bond-fraction_atoms_beyond_{_cutoff}_{_suffix}": round(
                number=len(atom_beyond_indices) / len(begin_dist_vector), ndigits=4
            ),
        }

        # Add data that depends on the presence of bonds
        _n_bonds = self.mol.GetNumBonds()
        if _n_bonds > 0:
            # Max and average distances to all bonds
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-bond-distance_to_all_bonds_mean"
            ] = float(round(number=np.mean(a=bond_dist_vector), ndigits=4))
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-bond-distance_to_all_bonds_max"
            ] = (
                float(round(number=np.max(a=bond_dist_vector), ndigits=4))
                if dimensionality == "3D"
                else int(np.max(a=bond_dist_vector))
            )

            # Within/beyond cutoff classifications bonds (index list)
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-bond-bonds_within_{_cutoff}_{_suffix}"
            ] = ",".join(map(str, bond_within_indices))

            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-bond-bonds_beyond_{_cutoff}_{_suffix}"
            ] = ",".join(map(str, bond_beyond_indices))

            # Within/beyond cutoff classifications bonds (count)
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-bond-n_bonds_within_{_cutoff}_{_suffix}"
            ] = len(bond_within_indices)

            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-bond-n_bonds_beyond_{_cutoff}_{_suffix}"
            ] = len(bond_beyond_indices)

            # Within/beyond cutoff classifications bonds (fraction)
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-bond-fraction_bonds_within_{_cutoff}_{_suffix}"
            ] = round(number=len(bond_within_indices) / self.mol.GetNumBonds(), ndigits=4)
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-bond-fraction_bonds_beyond_{_cutoff}_{_suffix}"
            ] = round(number=len(bond_beyond_indices) / self.mol.GetNumBonds(), ndigits=4)

        # Add data that is only defined for 2D molecules
        if dimensionality == "2D":
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-bond-atoms_{_cutoff}_{_suffix}_away"
            ] = ",".join(map(str, atom_exact_indices))
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-bond-n_atoms_{_cutoff}_{_suffix}_away"
            ] = len(atom_exact_indices)
            self.results[self.atom_bond_idx][
                f"bonafide{dimensionality}-bond-fraction_atoms_{_cutoff}_{_suffix}_away"
            ] = round(number=len(atom_exact_indices) / len(begin_dist_vector), ndigits=4)


class Bonafide2DBondAtomsBeyondXBonds(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "atoms_beyond_x_bonds", implemented within this
    package.

    The index of this feature is 37 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-atoms_beyond_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondAtomsWithinXBonds(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "atoms_within_x_bonds", implemented within this
    package.

    The index of this feature is 38 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-atoms_within_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondAtomsXBondsAway(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "atoms_x_bonds_away", implemented within this
    package.

    The index of this feature is 39 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-atoms_x_bonds_away`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondBondsBeyondXBonds(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "bonds_beyond_x_bonds", implemented within this
    package.

    The index of this feature is 40 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-bonds_beyond_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondBondsWithinXBonds(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "bonds_within_x_bonds", implemented within this
    package.

    The index of this feature is 41 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-bonds_within_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondFractionAtomsBeyondXBonds(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "fraction_atoms_beyond_x_bonds", implemented
    within this package.

    The index of this feature is 47 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-fraction_atoms_beyond_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondFractionAtomsWithinXBonds(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "fraction_atoms_within_x_bonds", implemented
    within this package.

    The index of this feature is 48 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-fraction_atoms_within_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondFractionAtomsXBondsAway(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "fraction_atoms_x_bonds_away", implemented within
    this package.

    The index of this feature is 49 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-fraction_atoms_x_bonds_away`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondFractionBondsBeyondXBonds(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "fraction_bonds_beyond_x_bonds", implemented
    within this package.

    The index of this feature is 50 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-fraction_bonds_beyond_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondFractionBondsWithinXBonds(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "fraction_bonds_within_x_bonds", implemented
    within this package.

    The index of this feature is 51 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-fraction_bonds_within_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondDistanceToAllAtomsMax(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "distance_to_all_atoms_max", implemented within
    this package.

    The index of this feature is 43 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-distance_to_all_atoms_max`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondDistanceToAllAtomsMean(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "distance_to_all_atoms_mean", implemented within
    this package.

    The index of this feature is 44 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-distance_to_all_atoms_mean`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondDistanceToAllBondsMax(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "distance_to_all_bonds_max", implemented within
    this package.

    The index of this feature is 45 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-distance_to_all_bonds_max`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondDistanceToAllBondsMean(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "distance_to_all_bonds_mean", implemented within
    this package.

    The index of this feature is 46 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-distance_to_all_bonds_mean`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondNAtomsBeyondXBonds(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "n_atoms_beyond_x_bonds", implemented within this
    package.

    The index of this feature is 53 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-n_atoms_beyond_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondNAtomsWithinXBonds(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "n_atoms_within_x_bonds", implemented within this
    package.

    The index of this feature is 54 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-n_atoms_within_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondNAtomsXBondsAway(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "n_atoms_x_bonds_away", implemented within this
    package.

    The index of this feature is 55 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-n_atoms_x_bonds_away`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondNBondsBeyondXBonds(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "n_bonds_beyond_x_bonds", implemented within this
    package.

    The index of this feature is 56 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-n_bonds_beyond_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide2DBondNBondsWithinXBonds(_BonafideBondDistance):
    """Feature factory for the 2D bond feature "n_bonds_within_x_bonds", implemented within this
    package.

    The index of this feature is 57 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-n_bonds_within_x_bonds`` feature."""
        self._run_calculation(dimensionality="2D")


class Bonafide3DBondAtomsBeyondXAngstrom(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "atoms_beyond_x_angstrom", implemented within
    this package.

    The index of this feature is 74 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-atoms_beyond_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondAtomsWithinXAngstrom(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "atoms_within_x_angstrom", implemented within
    this package.

    The index of this feature is 75 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-atoms_within_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondBondsBeyondXAngstrom(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "bonds_beyond_x_angstrom", implemented within
    this package.

    The index of this feature is 76 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-bonds_beyond_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondBondsWithinXAngstrom(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "bonds_within_x_angstrom", implemented within
    this package.

    The index of this feature is 77 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-bonds_within_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondDistanceToAllAtomsMax(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "distance_to_all_atoms_max", implemented within
    this package.

    The index of this feature is 78 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-distance_to_all_atoms_max`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondDistanceToAllAtomsMean(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "distance_to_all_atoms_mean", implemented within
    this package.

    The index of this feature is 79 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-distance_to_all_atoms_mean`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondDistanceToAllBondsMax(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "distance_to_all_bonds_max", implemented within
    this package.

    The index of this feature is 80 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-distance_to_all_bonds_max`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondDistanceToAllBondsMean(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "distance_to_all_bonds_mean", implemented within
    this package.

    The index of this feature is 81 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-distance_to_all_bonds_mean`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondFractionAtomsBeyondXAngstrom(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "fraction_atoms_beyond_x_angstrom", implemented
    within this package.

    The index of this feature is 82 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-fraction_atoms_beyond_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondFractionAtomsWithinXAngstrom(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "fraction_atoms_within_x_angstrom", implemented
    within this package.

    The index of this feature is 83 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-fraction_atoms_within_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondFractionBondsBeyondXAngstrom(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "fraction_bonds_beyond_x_angstrom", implemented
    within this package.

    The index of this feature is 84 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-fraction_bonds_beyond_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondFractionBondsWithinXAngstrom(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "fraction_bonds_within_x_angstrom", implemented
    within this package.

    The index of this feature is 85 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-fraction_bonds_within_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondNAtomsBeyondXAngstrom(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "n_atoms_beyond_x_angstrom", implemented within
    this package.

    The index of this feature is 86 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-n_atoms_beyond_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondNAtomsWithinXAngstrom(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "n_atoms_within_x_angstrom", implemented within
    this package.

    The index of this feature is 87 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-n_atoms_within_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondNBondsBeyondXAngstrom(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "n_bonds_beyond_x_angstrom", implemented within
    this package.

    The index of this feature is 88 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-n_bonds_beyond_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")


class Bonafide3DBondNBondsWithinXAngstrom(_BonafideBondDistance):
    """Feature factory for the 3D bond feature "n_bonds_within_x_angstrom", implemented within
    this package.

    The index of this feature is 89 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.distance" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide3D-bond-n_bonds_within_x_angstrom`` feature."""
        self._run_calculation(dimensionality="3D")
