"""Dispersion features from ``MORFEUS``."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from morfeus import Dispersion

from bonafide.utils.base_featurizer import BaseFeaturizer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _Morfeus3DAtomDispersion(BaseFeaturizer):
    """Parent feature factory for the 3D atom MORFEUS dispersion features.

    For details, please refer to the MORFEUS documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    09.09.2025).
    """

    dispersion_: Dispersion

    density: float
    excluded_atoms: Optional[List[int]]
    included_atoms: Optional[List[int]]
    radii: Optional[Union[List[float], NDArray[np.float64]]]
    radii_type: str

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def _run_morfeus(self) -> bool:
        """Run MORFEUS and populate the dispersion attribute (``dispersion_``).

        Returns
        -------
        bool
            Whether MORFEUS ran (successfully).
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

        if self.included_atoms == []:
            self.included_atoms = None
        else:
            assert self.included_atoms is not None  # for type checker
            self.included_atoms = self._validate_atom_indices(
                atom_indices_list=self.included_atoms,
                parameter_name="included_atoms",
                all_indices=_atom_indices,
            )
            if self.included_atoms is None:
                return False

        # Run MORFEUS
        assert self.coordinates is not None  # for type checker
        self.dispersion_ = Dispersion(
            elements=self.elements,
            coordinates=self.coordinates,
            radii=self.radii,
            radii_type=self.radii_type,
            density=self.density,
            excluded_atoms=self.excluded_atoms,
            included_atoms=self.included_atoms,
        )

        # Save data
        with open(f"{self.__class__.__name__}__{self.conformer_name}.pkl", "wb") as f:
            pickle.dump(self.dispersion_, f)

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


class Morfeus3DAtomPInt(_Morfeus3DAtomDispersion):
    """Feature factory for the 3D atom feature "p_int", calculated with morfeus.

    The index of this feature is 188 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.dispersion" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-p_int`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            for atom_idx, value in self.dispersion_.atom_p_int.items():
                self.results[atom_idx - 1] = {
                    self.feature_name: float(value)
                }  # morfeus is 1-indexed


class Morfeus3DAtomPMax(_Morfeus3DAtomDispersion):
    """Feature factory for the 3D atom feature "p_max", calculated with morfeus.

    The index of this feature is 189 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.dispersion" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-p_max`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            for atom_idx, value in self.dispersion_.atom_p_max.items():
                self.results[atom_idx - 1] = {
                    self.feature_name: float(value)
                }  # morfeus is 1-indexed


class Morfeus3DAtomPMin(_Morfeus3DAtomDispersion):
    """Feature factory for the 3D atom feature "p_min", calculated with morfeus.

    The index of this feature is 190 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.dispersion" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-p_min`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            for atom_idx, value in self.dispersion_.atom_p_min.items():
                self.results[atom_idx - 1] = {
                    self.feature_name: float(value)
                }  # morfeus is 1-indexed
