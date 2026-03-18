"""Solid angle features from ``MORFEUS``."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from morfeus import SolidAngle

from bonafide.utils.base_featurizer import BaseFeaturizer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _Morfeus3DAtomSolidAngle(BaseFeaturizer):
    """Parent feature factory for the 3D atom MORFEUS solid angle features.

    For details, please refer to the MORFEUS documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    09.09.2025).
    """

    sa_: SolidAngle

    density: float
    radii: Optional[Union[List[float], NDArray[np.float64]]]
    radii_type: str

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _run_morfeus(self) -> None:
        """Run MORFEUS and populate the solid angle attribute (``sa_``).

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
        self.sa_ = SolidAngle(
            elements=self.elements,
            coordinates=self.coordinates,
            metal_index=self.atom_bond_idx + 1,  # MORFEUS is 1-indexed
            radii=self.radii,
            radii_type=self.radii_type,
            density=self.density,
        )

        # Save data
        with open(f"{self.__class__.__name__}__{self.conformer_name}.pkl", "wb") as f:
            pickle.dump(self.sa_, f)


class Morfeus3DAtomConeAngleSolid(_Morfeus3DAtomSolidAngle):
    """Feature factory for the 3D atom feature "cone_angle_solid", calculated with morfeus.

    The index of this feature is 171 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.cone_and_solid_angle" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-cone_angle_solid`` feature."""
        self._run_morfeus()
        self.results[self.atom_bond_idx] = {self.feature_name: float(self.sa_.cone_angle)}


class Morfeus3DAtomConeAngleSolidGParameter(_Morfeus3DAtomSolidAngle):
    """Feature factory for the 3D atom feature "cone_angle_solid_g_parameter", calculated with
    morfeus.

    The index of this feature is 172 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.cone_and_solid_angle" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-cone_angle_solid_g_parameter`` feature."""
        self._run_morfeus()
        self.results[self.atom_bond_idx] = {self.feature_name: float(self.sa_.G)}
