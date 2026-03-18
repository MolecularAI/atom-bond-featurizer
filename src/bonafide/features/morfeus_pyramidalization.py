"""Pyramidalization features from ``MORFEUS``."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from morfeus import Pyramidalization

from bonafide.utils.base_featurizer import BaseFeaturizer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _Morfeus3DAtomPyramidalization(BaseFeaturizer):
    """Parent feature factory for the 3D atom MORFEUS pyramidalization features.

    For details, please refer to the MORFEUS documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    09.09.2025).
    """

    pyra_: Pyramidalization

    excluded_atoms: Optional[List[int]]
    method: str
    radii: Optional[Union[List[float], NDArray[np.float64]]]
    scale_factor: float

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _run_morfeus(self) -> bool:
        """Run MORFEUS and populate the pyramidalization object (``pyra_``).

        Returns
        -------
        bool
            Returns ``True`` if MORFEUS ran successfully and the data is ready to be saved, and
            ``False`` if there was an error and the data should not be saved.
        """
        # Modify the user input if necessary to comply with MORFEUS requirements
        if self.radii == []:
            self.radii = None
        else:
            self.radii = np.array(self.radii)

        _atom_indices = list(range(len(self.elements)))

        if self.excluded_atoms == []:
            self.excluded_atoms = None
        else:
            assert self.excluded_atoms is not None  # for type checker
            self.excluded_atoms = self._validate_atom_indices(
                atom_indices_list=self.excluded_atoms,
                parameter_name="excluded_atoms",
                all_indices=_atom_indices,
            )
            if self.excluded_atoms is None:
                return False

        # Run MORFEUS
        assert self.coordinates is not None  # for type checker
        self.pyra_ = Pyramidalization(
            elements=self.elements,
            coordinates=self.coordinates,
            atom_index=self.atom_bond_idx + 1,  # MORFEUS is 1-indexed
            radii=self.radii,
            excluded_atoms=self.excluded_atoms,
            method=self.method,
            scale_factor=self.scale_factor,
        )

        # Save data
        with open(f"{self.__class__.__name__}__{self.conformer_name}.pkl", "wb") as f:
            pickle.dump(self.pyra_, f)

        return True

    def _validate_atom_indices(
        self, atom_indices_list: List[int], parameter_name: str, all_indices: List[int]
    ) -> Optional[List[int]]:
        """Validate user-provided atom indices.

        Parameters
        ----------
        atom_indices_list : List[int]
            The list of atom indices to be validated.
        parameter_name : str
            The name of the parameter being validated (for error messages).
        all_indices : List[int]
            A list of all valid atom indices.

        Returns
        -------
        Optional[List[int]]
            Returns the validated list of atom indices (converted to 1-indexed) or ``None`` if
            validation fails.
        """
        for idx in atom_indices_list:
            if idx not in all_indices:
                self._err = (
                    f"Invalid input to '{parameter_name}': atom index {idx} is out of range."
                )
                return None

        atom_indices_list = [idx + 1 for idx in atom_indices_list]  # MORFEUS is 1-indexed
        return atom_indices_list


class Morfeus3DAtomPyramidalizationAlpha(_Morfeus3DAtomPyramidalization):
    """Feature factory for the 3D atom feature "pyramidalization_alpha", calculated with
    morfeus.

    The index of this feature is 191 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.pyramidalization" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-pyramidalization_alpha`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            self.results[self.atom_bond_idx] = {self.feature_name: float(self.pyra_.alpha)}


class Morfeus3DAtomPyramidalizationAlphas(_Morfeus3DAtomPyramidalization):
    """Feature factory for the 3D atom feature "pyramidalization_alphas", calculated with
    morfeus.

    The index of this feature is 192 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.pyramidalization" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-pyramidalization_alphas`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            self.results[self.atom_bond_idx] = {
                self.feature_name: ",".join(map(str, self.pyra_.alphas))
            }


class Morfeus3DAtomPyramidalizationGavrish(_Morfeus3DAtomPyramidalization):
    """Feature factory for the 3D atom feature "pyramidalization_gavrish", calculated with
    morfeus.

    The index of this feature is 193 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.pyramidalization" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-pyramidalization_gavrish`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            self.results[self.atom_bond_idx] = {self.feature_name: float(self.pyra_.P_angle)}


class Morfeus3DAtomPyramidalizationNeighborIndices(_Morfeus3DAtomPyramidalization):
    """Feature factory for the 3D atom feature "pyramidalization_neighbor_indices", calculated
    with morfeus.

    The index of this feature is 194 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.pyramidalization" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-pyramidalization_neighbor_indices`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            self.results[self.atom_bond_idx] = {
                self.feature_name: ",".join(
                    map(str, [idx - 1 for idx in self.pyra_.neighbor_indices])
                )
            }


class Morfeus3DAtomPyramidalizationRadhakrishnan(_Morfeus3DAtomPyramidalization):
    """Feature factory for the 3D atom feature "pyramidalization_radhakrishnan", calculated with
    morfeus.

    The index of this feature is 195 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.pyramidalization" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-pyramidalization_radhakrishnan`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            self.results[self.atom_bond_idx] = {self.feature_name: float(self.pyra_.P)}
