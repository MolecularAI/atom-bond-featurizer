"""Cone angle features from ``MORFEUS``."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from morfeus import ConeAngle

from bonafide.utils.base_featurizer import BaseFeaturizer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _Morfeus3DAtomConeAngle(BaseFeaturizer):
    """Parent feature factory for the 3D atom MORFEUS cone angle features.

    For details, please refer to the MORFEUS documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    09.09.2025).
    """

    ca_: ConeAngle

    density: float
    radii: Optional[Union[List[float], NDArray[np.float64]]]
    radii_type: str

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _run_morfeus(self) -> None:
        """Run MORFEUS and populate the cone angle attribute (``ca_``).

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
        self.ca_ = ConeAngle(
            elements=self.elements,
            coordinates=self.coordinates,
            atom_1=self.atom_bond_idx + 1,  # MORFEUS is 1-indexed
            radii=self.radii,
            radii_type=self.radii_type,
            method="internal",
        )

        # Save data
        with open(f"{self.__class__.__name__}__{self.conformer_name}.pkl", "wb") as f:
            pickle.dump(self.ca_, f)


class Morfeus3DAtomConeAngle(_Morfeus3DAtomConeAngle):
    """Feature factory for the 3D atom feature "cone_angle", calculated with morfeus.

    The index of this feature is 170 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.cone_and_solid_angle" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-cone_angle`` feature."""
        self._run_morfeus()
        self.results[self.atom_bond_idx] = {self.feature_name: float(self.ca_.cone_angle)}


class Morfeus3DAtomConeTangentAtoms(_Morfeus3DAtomConeAngle):
    """Feature factory for the 3D atom feature "cone_tangent_atoms", calculated with morfeus.

    The index of this feature is 173 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.cone_and_solid_angle" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-cone_tangent_atoms`` feature."""
        self._run_morfeus()
        self.results[self.atom_bond_idx] = {
            self.feature_name: ",".join(map(str, [idx - 1 for idx in self.ca_.tangent_atoms]))
        }
