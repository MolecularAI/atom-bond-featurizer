"""Buried volume features from ``MORFEUS``."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from morfeus import BuriedVolume

from bonafide.utils.base_featurizer import BaseFeaturizer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _Morfeus3DAtomBV(BaseFeaturizer):
    """Parent feature factory for the 3D atom MORFEUS buried volume features.

    For details, please refer to the MORFEUS documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    09.09.2025).
    """

    bv_: BuriedVolume

    density: float
    distal_volume_method: str
    distal_volume_sasa_density: float
    excluded_atoms: Optional[List[int]]
    include_hs: bool
    radii: Optional[Union[List[float], NDArray[np.float64]]]
    radii_scale: float
    radii_type: str
    radius: float
    xz_plane_atoms: Optional[List[int]]
    z_axis_atoms: Optional[List[int]]

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _run_morfeus(self, do_octant_analysis: bool = False) -> bool:
        """Run MORFEUS and populate the buried volume attribute (``bv_``).

        Parameters
        ----------
        do_octant_analysis : bool, optional
            Whether to calculate the decomposition of the buried-volume features in octants and
            quadrants, by default ``False``.

        Returns
        -------
        bool
            Whether morfeus ran (successfully).
        """
        # Modify the user input if necessary to comply with morfeus requirements
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

        if self.z_axis_atoms == []:
            self.z_axis_atoms = None
        else:
            assert self.z_axis_atoms is not None  # for type checker
            self.z_axis_atoms = self._validate_atom_indices(
                atom_indices_list=self.z_axis_atoms,
                parameter_name="z_axis_atoms",
                all_indices=_atom_indices,
            )
            if self.z_axis_atoms is None:
                return False

        if self.xz_plane_atoms == []:
            self.xz_plane_atoms = None
        else:
            assert self.xz_plane_atoms is not None  # for type checker
            self.xz_plane_atoms = self._validate_atom_indices(
                atom_indices_list=self.xz_plane_atoms,
                parameter_name="xz_plane_atoms",
                all_indices=_atom_indices,
            )
            if self.xz_plane_atoms is None:
                return False

        # Run morfeus
        assert self.coordinates is not None  # for type checker
        self.bv_ = BuriedVolume(
            elements=self.elements,
            coordinates=self.coordinates,
            metal_index=self.atom_bond_idx + 1,  # morfeus is 1-indexed
            excluded_atoms=self.excluded_atoms,
            radii=self.radii,
            include_hs=self.include_hs,
            radius=self.radius,
            radii_type=self.radii_type,
            radii_scale=self.radii_scale,
            density=self.density,
            z_axis_atoms=self.z_axis_atoms,
            xz_plane_atoms=self.xz_plane_atoms,
        )

        # Perform octant analysis if requested
        if do_octant_analysis is True:
            self.bv_.octant_analysis()

        # Save data
        with open(f"{self.__class__.__name__}__{self.conformer_name}.pkl", "wb") as f:
            pickle.dump(self.bv_, f)

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


class Morfeus3DAtomBuriedVolume(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "buried_volume", calculated with morfeus.

    The index of this feature is 167 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-buried_volume`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            self.results[self.atom_bond_idx] = {self.feature_name: self.bv_.buried_volume}


class Morfeus3DAtomBuriedVolumeOctants(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "buried_volume_octants", calculated with morfeus.

    The index of this feature is 168 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-buried_volume_octants`` feature."""
        _save_data = self._run_morfeus(do_octant_analysis=True)
        if _save_data is True:
            val = ",".join([str(v) for v in self.bv_.octants["buried_volume"].values()])
            self.results[self.atom_bond_idx] = {self.feature_name: val}


class Morfeus3DAtomBuriedVolumeQuadrants(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "buried_volume_quadrants", calculated with
    morfeus.

    The index of this feature is 169 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-buried_volume_quadrants`` feature."""
        _save_data = self._run_morfeus(do_octant_analysis=True)
        if _save_data is True:
            val = ",".join([str(v) for v in self.bv_.quadrants["buried_volume"].values()])
            self.results[self.atom_bond_idx] = {self.feature_name: val}


class Morfeus3DAtomDistalVolume(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "distal_volume", calculated with morfeus.

    The index of this feature is 174 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-distal_volume`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            self.bv_.compute_distal_volume(
                method=self.distal_volume_method,
                octants=False,
                sasa_density=self.distal_volume_sasa_density,
            )
            val = float(self.bv_.distal_volume)
            if val < 0:
                val = 0.0
            self.results[self.atom_bond_idx] = {self.feature_name: val}

            # Save data
            with open(f"{self.__class__.__name__}__{self.conformer_name}.pkl", "wb") as f:
                pickle.dump(self.bv_, f)


class Morfeus3DAtomDistalVolumeOctants(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "distal_volume_octants", calculated with morfeus.

    The index of this feature is 175 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-distal_volume_octants`` feature."""
        _save_data = self._run_morfeus(do_octant_analysis=True)
        if _save_data is True:
            self.bv_.compute_distal_volume(
                method="buried_volume",
                octants=True,
                sasa_density=self.distal_volume_sasa_density,
            )
            vals = [0.0 if v < 0 else v for v in self.bv_.octants["distal_volume"].values()]
            val = ",".join([str(v) for v in vals])
            self.results[self.atom_bond_idx] = {self.feature_name: val}

            # Save data
            with open(f"{self.__class__.__name__}__{self.conformer_name}.pkl", "wb") as f:
                pickle.dump(self.bv_, f)


class Morfeus3DAtomDistalVolumeQuadrants(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "distal_volume_quadrants", calculated with
    morfeus.

    The index of this feature is 176 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-distal_volume_quadrants`` feature."""
        _save_data = self._run_morfeus(do_octant_analysis=True)
        if _save_data is True:
            self.bv_.compute_distal_volume(
                method="buried_volume",
                octants=True,
                sasa_density=self.distal_volume_sasa_density,
            )
            vals = [0.0 if v < 0 else v for v in self.bv_.quadrants["distal_volume"].values()]
            val = ",".join([str(v) for v in vals])
            self.results[self.atom_bond_idx] = {self.feature_name: val}

            # Save data
            with open(f"{self.__class__.__name__}__{self.conformer_name}.pkl", "wb") as f:
                pickle.dump(self.bv_, f)


class Morfeus3DAtomFractionBuriedVolume(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "fraction_buried_volume", calculated with
    morfeus.

    The index of this feature is 177 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-fraction_buried_volume`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            self.results[self.atom_bond_idx] = {self.feature_name: self.bv_.fraction_buried_volume}


class Morfeus3DAtomFractionBuriedVolumeOctants(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "fraction_buried_volume_octants", calculated with
    morfeus.

    The index of this feature is 178 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-fraction_buried_volume_octants`` feature."""
        _save_data = self._run_morfeus(do_octant_analysis=True)
        if _save_data is True:
            val = ",".join(
                [str(v / 100) for v in self.bv_.octants["percent_buried_volume"].values()]
            )
            self.results[self.atom_bond_idx] = {self.feature_name: val}


class Morfeus3DAtomFractionBuriedVolumeQuadrants(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "fraction_buried_volume_quadrants", calculated
    with morfeus.

    The index of this feature is 179 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-fraction_buried_volume_quadrants`` feature."""
        _save_data = self._run_morfeus(do_octant_analysis=True)
        if _save_data is True:
            val = ",".join(
                [str(v / 100) for v in self.bv_.quadrants["percent_buried_volume"].values()]
            )
            self.results[self.atom_bond_idx] = {self.feature_name: val}


class Morfeus3DAtomFractionFreeVolume(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "fraction_free_volume", calculated with morfeus.

    The index of this feature is 180 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-fraction_free_volume`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            self.results[self.atom_bond_idx] = {
                self.feature_name: 1 - self.bv_.fraction_buried_volume
            }


class Morfeus3DAtomFractionFreeVolumeOctants(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "fraction_free_volume_octants", calculated with
    morfeus.

    The index of this feature is 181 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-fraction_free_volume_octants`` feature."""
        _save_data = self._run_morfeus(do_octant_analysis=True)
        if _save_data is True:
            val = ",".join(
                [str(1 - v / 100) for v in self.bv_.octants["percent_buried_volume"].values()]
            )
            self.results[self.atom_bond_idx] = {self.feature_name: val}


class Morfeus3DAtomFractionFreeVolumeQuadrants(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "fraction_free_volume_quadrants", calculated with
    morfeus.

    The index of this feature is 182 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-fraction_free_volume_quadrants`` feature."""
        _save_data = self._run_morfeus(do_octant_analysis=True)
        if _save_data is True:
            val = ",".join(
                [str(1 - v / 100) for v in self.bv_.quadrants["percent_buried_volume"].values()]
            )
            self.results[self.atom_bond_idx] = {self.feature_name: val}


class Morfeus3DAtomFreeVolume(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "free_volume", calculated with morfeus.

    The index of this feature is 183 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-free_volume`` feature."""
        _save_data = self._run_morfeus()
        if _save_data is True:
            self.results[self.atom_bond_idx] = {self.feature_name: self.bv_.free_volume}


class Morfeus3DAtomFreeVolumeOctants(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "free_volume_octants", calculated with morfeus.

    The index of this feature is 184 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-free_volume_octants`` feature."""
        _save_data = self._run_morfeus(do_octant_analysis=True)
        if _save_data is True:
            val = ",".join([str(v) for v in self.bv_.octants["free_volume"].values()])
            self.results[self.atom_bond_idx] = {self.feature_name: val}


class Morfeus3DAtomFreeVolumeQuadrants(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "free_volume_quadrants", calculated with morfeus.

    The index of this feature is 185 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-free_volume_quadrants`` feature."""
        _save_data = self._run_morfeus(do_octant_analysis=True)
        if _save_data is True:
            val = ",".join([str(v) for v in self.bv_.quadrants["free_volume"].values()])
            self.results[self.atom_bond_idx] = {self.feature_name: val}


class Morfeus3DAtomMolecularVolumeOctants(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "molecular_volume_octants", calculated with
    morfeus.

    The index of this feature is 186 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-molecular_volume_octants`` feature."""
        _save_data = self._run_morfeus(do_octant_analysis=True)
        if _save_data is True:
            self.bv_.compute_distal_volume(
                method="buried_volume",
                octants=True,
                sasa_density=self.distal_volume_sasa_density,
            )
            val = ",".join([str(v) for v in self.bv_.octants["molecular_volume"].values()])
            self.results[self.atom_bond_idx] = {self.feature_name: val}

            # Save data
            with open(f"{self.__class__.__name__}__{self.conformer_name}.pkl", "wb") as f:
                pickle.dump(self.bv_, f)


class Morfeus3DAtomMolecularVolumeQuadrants(_Morfeus3DAtomBV):
    """Feature factory for the 3D atom feature "molecular_volume_quadrants", calculated with
    morfeus.

    The index of this feature is 187 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.buried_volume" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-atom-molecular_volume_quadrants`` feature."""
        _save_data = self._run_morfeus(do_octant_analysis=True)
        if _save_data is True:
            self.bv_.compute_distal_volume(
                method="buried_volume",
                octants=True,
                sasa_density=self.distal_volume_sasa_density,
            )
            val = ",".join([str(v) for v in self.bv_.quadrants["molecular_volume"].values()])
            self.results[self.atom_bond_idx] = {self.feature_name: val}

            # Save data
            with open(f"{self.__class__.__name__}__{self.conformer_name}.pkl", "wb") as f:
                pickle.dump(self.bv_, f)
