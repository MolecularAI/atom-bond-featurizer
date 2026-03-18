"""Solvent-accessible surface features from ``MORFEUS``."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from morfeus import SASA

from bonafide.utils.base_featurizer import BaseFeaturizer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _Morfeus3DAtomSASA(BaseFeaturizer):
    """Parent feature factory for the 3D atom MORFEUS SASA features.

    For details, please refer to the MORFEUS documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    09.09.2025).
    """

    sasa_: SASA

    density: float
    probe_radius: float
    radii: Optional[Union[List[float], NDArray[np.float64]]]
    radii_type: str

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def _run_morfeus(self) -> None:
        """Run MORFEUS and populate the solvent-accessible surface area attribute (``sasa_``).

        Returns
        -------
        None
        """
        # Modify the user input if necessary to comply with MORFEUS requirements
        if self.radii == []:
            self.radii = None
        else:
            self.radii = np.array(self.radii)

        # Run MORFEUS
        assert self.coordinates is not None  # for type checker
        self.sasa_ = SASA(
            elements=self.elements,
            coordinates=self.coordinates,
            radii=self.radii,
            radii_type=self.radii_type,
            probe_radius=self.probe_radius,
            density=self.density,
        )

        # Save data
        with open(f"{self.__class__.__name__}__{self.conformer_name}.pkl", "wb") as f:
            pickle.dump(self.sasa_, f)


class Morfeus3DAtomSasAtomArea(_Morfeus3DAtomSASA):
    """Feature factory for the 3D atom feature "sas_atom_area", calculated with morfeus.

    The index of this feature is 196 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.sasa" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-sas_atom_area`` feature."""
        self._run_morfeus()
        for atom_idx, value in self.sasa_.atom_areas.items():
            self.results[atom_idx - 1] = {self.feature_name: float(value)}  # morfeus is 1-indexed


class Morfeus3DAtomSasFractionAtomArea(_Morfeus3DAtomSASA):
    """Feature factory for the 3D atom feature "sas_fraction_atom_area", calculated with
    morfeus.

    The index of this feature is 197 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.sasa" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-sas_fraction_atom_area`` feature."""
        self._run_morfeus()
        for atom_idx, value in self.sasa_.atom_areas.items():
            ratio = value / self.sasa_.area
            self.results[atom_idx - 1] = {self.feature_name: float(ratio)}  # morfeus is 1-indexed
