"""Autocorrelation features for atoms in 2D molecules."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from rdkit import Chem

from bonafide.utils.base_featurizer import BaseFeaturizer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _Bonafide2DAtomAutocorrelation(BaseFeaturizer):
    """Parent feature factory for the 2D atom autocorrelation features."""

    depth: int
    iterable_option: str

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def run_workflow(self, operant: Any, scale: bool) -> None:
        """Execute the workflow for calculating an autocorrelation feature vector for a give root
        atom.

        Parameters
        ----------
        operant : Any
            The mathematical operation (sum, difference, product, mean, abs difference) to be
            applied between the property of the root atom and the respective other atom.
        scale : bool
            Whether to scale the autocorrelation feature at each depth by the number of atoms at
            that depth.

        Returns
        -------
        None
        """
        # Get the distance matrix, either from the cache or by calculating it
        _feature_name = "rdkit2d-global-bond_distance_matrix"
        if _feature_name not in self.global_feature_cache[self.conformer_idx]:
            distance_matrix = Chem.GetDistanceMatrix(self.mol)
            self.global_feature_cache[self.conformer_idx][_feature_name] = distance_matrix
        else:
            distance_matrix = self.global_feature_cache[self.conformer_idx][_feature_name]

        # Calculate autocorrelation vector
        dist_vector = distance_matrix[self.atom_bond_idx]

        property_vector = self._get_property_vector(dist_vector=dist_vector)
        if property_vector is None:
            return

        if "_inaccessible" in property_vector:
            self._err = (
                f"the property vector for the '{self.iterable_option}' feature contains "
                f"inaccessible values. Therefore, the requested autocorrelation feature cannot "
                "be calculated"
            )
            return

        autocorr_vector = self._get_autocorrelation_vector(
            dist_vector=dist_vector, property_vector=property_vector, operant=operant, scale=scale
        )

        # Modify feature_name to also include the name of the underlying atom feature
        f_name = self.iterable_option.split("-")[-1]
        self.feature_name = f"{self.feature_name}__{f_name}"

        # Safe results to the results dictionary
        self.results[self.atom_bond_idx] = {self.feature_name: autocorr_vector}

    def _get_property_vector(
        self, dist_vector: NDArray[np.float64]
    ) -> Optional[NDArray[np.float64]]:
        """Get the atom property vector for the requested feature to be used in the generation of
        the autocorrelation features.

        This property must be precomputed for every atom in the molecule, requested by the
        ``iterable_option`` input available in the configuration settings of the autocorrelation
        feature.

        Parameters
        ----------
        dist_vector : NDArray[np.float64]
            The vector of the shortest distances in bonds from the root atom to all other atoms
            in the molecule.

        Returns
        -------
        Optional[NDArray[np.float64]]
            The property vector for the requested feature for all atoms in the molecule or ``None``
            if an error occurred.
        """
        # Check if feature for autocorrelation was precomputed
        if self.iterable_option not in self.feature_cache[self.conformer_idx]:
            self._err = (
                f"the '{self.iterable_option}' feature that was requested to be used to "
                "calculate the autocorrelation vector was not precomputed for conformer "
                f"with index {self.conformer_idx}. Calculate this feature before requesting "
                "the autocorrelation feature"
            )
            return None

        # Get the property vector
        _cached_data = self.feature_cache[self.conformer_idx][self.iterable_option]
        _n_atoms = self.mol.GetNumAtoms()
        property_vector: List[Any] = ["_missing" for _ in range(_n_atoms)]
        for idx, value in _cached_data.items():
            property_vector[idx] = value

        # Check if a required value from the property vector is missing
        for idx, dist in enumerate(dist_vector):
            if dist > self.depth:
                continue
            if property_vector[idx] == "_missing":
                self._err = (
                    f"the property vector for the '{self.iterable_option}' feature is missing "
                    f"a value at atom index {idx}. Therefore, the requested autocorrelation "
                    "feature cannot be calculated"
                )
                return None

        return np.array(property_vector)

    def _get_autocorrelation_vector(
        self,
        dist_vector: NDArray[np.float64],
        property_vector: NDArray[np.float64],
        operant: Any,
        scale: bool,
    ) -> str:
        """Calculate the autocorrelation vector.

        Parameters
        ----------
        dist_vector : NDArray[np.float64]
            The vector of the shortest distances in bonds from the root atom to all other atoms
            in the molecule.
        property_vector : NDArray[np.float64]
            The property vector for the requested feature for all atoms in the molecule.
        operant : Any
            The mathematical operation (sum, difference, product, mean, abs difference) to be
            applied between the property of the root atom and the respective other atom.
        scale : bool
            Whether to scale the autocorrelation feature at each depth by the number of atoms at
            that depth.

        Returns
        -------
        str
            The autocorrelation vector as a comma-separated string.
        """
        root_prop = property_vector[self.atom_bond_idx]
        autocorr_vector = []
        for d in range(self.depth + 1):
            kronecker_vec = self._kronecker_delta(arr=dist_vector, target_value=d)
            value = np.sum(operant(root_prop, property_vector) * kronecker_vec)
            if scale is True:
                value /= np.count_nonzero(a=kronecker_vec)
            autocorr_vector.append(value)
        autocorr_vector = [float(x) for x in autocorr_vector]
        autocorr_vector_str = ",".join(
            [str(round(number=val, ndigits=8)) for val in autocorr_vector]
        )
        return autocorr_vector_str

    @staticmethod
    def _kronecker_delta(arr: NDArray[np.float64], target_value: int) -> NDArray[np.float64]:
        """Calculate the Kronecker delta for a given array and target value.

        Parameters
        ----------
        arr : NDArray[np.float64]
            The input array (i.e., the distance vector of a given atom to all atoms).
        target_value : int
            The target value (i.e., the current depth).

        Returns
        -------
        NDArray[np.float64]
            The Kronecker delta array, where elements equal to the target value are 1.0, and all
            other elements are 0.0.
        """
        res: NDArray[np.float64] = (arr == target_value).astype(float)
        return res

    @staticmethod
    def _mean(num: float, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate the mean between a number and an array.

        Parameters
        ----------
        num : float
            The number (i.e., the property of the root atom).
        arr : NDArray[np.float64]
            The array (i.e., the property vector of all atoms in the molecule).

        Returns
        -------
        NDArray[np.float64]
            The mean between the number and each element in the array.
        """
        return (num + arr) / 2

    @staticmethod
    def _abs_diff(num: float, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate the absolute difference between a number and an array.

        Parameters
        ----------
        num : float
            The number (i.e., the property of the root atom).
        arr : NDArray[np.float64]
            The array (i.e., the property vector of all atoms in the molecule).

        Returns
        -------
        NDArray[np.float64]
            The absolute difference between the number and each element in the array.
        """
        return np.abs(num - arr)


class Bonafide2DAtomAutocorrelationAbsDifference(_Bonafide2DAtomAutocorrelation):
    """Feature factory for the 2D atom feature "autocorrelation_abs_difference", implemented
    within this package.

    The index of this feature is 5 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.autocorrelation" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-autocorrelation_abs_difference`` feature."""
        self.run_workflow(operant=self._abs_diff, scale=False)


class Bonafide2DAtomAutocorrelationDifference(_Bonafide2DAtomAutocorrelation):
    """Feature factory for the 2D atom feature "autocorrelation_difference", implemented within
    this package.

    The index of this feature is 6 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.autocorrelation" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-autocorrelation_difference`` feature."""
        self.run_workflow(operant=operator.sub, scale=False)


class Bonafide2DAtomAutocorrelationMean(_Bonafide2DAtomAutocorrelation):
    """Feature factory for the 2D atom feature "autocorrelation_mean", implemented within this
    package.

    The index of this feature is 7 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.autocorrelation" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-autocorrelation_mean`` feature."""
        self.run_workflow(operant=self._mean, scale=False)


class Bonafide2DAtomAutocorrelationProduct(_Bonafide2DAtomAutocorrelation):
    """Feature factory for the 2D atom feature "autocorrelation_product", implemented within
    this package.

    The index of this feature is 8 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.autocorrelation" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-autocorrelation_product`` feature."""
        self.run_workflow(operant=operator.mul, scale=False)


class Bonafide2DAtomAutocorrelationScaledAbsDifference(_Bonafide2DAtomAutocorrelation):
    """Feature factory for the 2D atom feature "autocorrelation_scaled_abs_difference",
    implemented within this package.

    The index of this feature is 9 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.autocorrelation" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-autocorrelation_scaled_abs_difference`` feature."""
        self.run_workflow(operant=self._abs_diff, scale=True)


class Bonafide2DAtomAutocorrelationScaledDifference(_Bonafide2DAtomAutocorrelation):
    """Feature factory for the 2D atom feature "autocorrelation_scaled_difference", implemented
    within this package.

    The index of this feature is 10 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.autocorrelation" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-autocorrelation_scaled_difference`` feature."""
        self.run_workflow(operant=operator.sub, scale=True)


class Bonafide2DAtomAutocorrelationScaledMean(_Bonafide2DAtomAutocorrelation):
    """Feature factory for the 2D atom feature "autocorrelation_scaled_mean", implemented within
    this package.

    The index of this feature is 11 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.autocorrelation" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-autocorrelation_scaled_mean`` feature."""
        self.run_workflow(operant=self._mean, scale=True)


class Bonafide2DAtomAutocorrelationScaledProduct(_Bonafide2DAtomAutocorrelation):
    """Feature factory for the 2D atom feature "autocorrelation_scaled_product", implemented
    within this package.

    The index of this feature is 12 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.autocorrelation" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-autocorrelation_scaled_product`` feature."""
        self.run_workflow(operant=operator.mul, scale=True)


class Bonafide2DAtomAutocorrelationScaledSum(_Bonafide2DAtomAutocorrelation):
    """Feature factory for the 2D atom feature "autocorrelation_scaled_sum", implemented within
    this package.

    The index of this feature is 13 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.autocorrelation" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-autocorrelation_scaled_sum`` feature."""
        self.run_workflow(operant=operator.add, scale=True)


class Bonafide2DAtomAutocorrelationSum(_Bonafide2DAtomAutocorrelation):
    """Feature factory for the 2D atom feature "autocorrelation_sum", implemented within this
    package.

    The index of this feature is 14 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.autocorrelation" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-autocorrelation_sum`` feature."""
        self.run_workflow(operant=operator.add, scale=False)
