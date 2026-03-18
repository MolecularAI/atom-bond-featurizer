"""Helper methods for calculating C-DFT redox descriptors."""

from typing import Dict, List, Optional, Tuple, Union, cast

from bonafide.utils.global_properties import (
    calculate_global_cdft_descriptors_redox,
)


class CdftLocalRedoxMixin:
    """Mixin class to provide functionality required for calculating local C-DFT descriptors based
    on the ionization potential and electron affinity.

    Attributes
    ----------
    conformer_idx : int
        The index of the conformer in the molecule vault.
    energy_n : Tuple[Optional[float], str]
        The energy of the actual molecule that was calculated or provided by the user as value
        unit pair. The first entry of the tuple is ``None`` if the energy data is not available.
    energy_n_minus1 : Tuple[Optional[float], str]
        The energy of the one-electron-oxidized molecule that was calculated or provided by the
        user as value unit pair. The first entry of the tuple is ``None`` if the energy data is not
        available.
    energy_n_plus1 : Tuple[Optional[float], str]
        The energy of the one-electron-reduced molecule that was calculated or provided by the
        user as value unit pair. The first entry of the tuple is ``None`` if the energy data is not
        available.
    global_feature_cache : List[Dict[str, Optional[Union[str, bool, int, float]]]]
        The cache of global features for each conformer. The individual list entries are
        dictionaries with the feature names as keys and feature values as values.
    """

    conformer_idx: int
    energy_n: Tuple[Optional[float], str]
    energy_n_minus1: Tuple[Optional[float], str]
    energy_n_plus1: Tuple[Optional[float], str]
    global_feature_cache: List[Dict[str, Optional[Union[str, bool, int, float]]]]

    def _check_energy_data(self) -> Optional[str]:
        """Check if the required energy data is available for all three redox states.

        Returns
        -------
        Optional[str]
            An error message if any of the required energy data is missing, otherwise ``None``.
        """
        _errmsg = None
        _state_map = {"n": self.energy_n, "n-1": self.energy_n_minus1, "n+1": self.energy_n_plus1}

        for state_id, energy in _state_map.items():
            if energy is None:
                _errmsg = (
                    f"for requesting data from '{self.__class__.__name__}', energy data for all "
                    "three redox states (actual molecule, one-electron-reduced, "
                    "one-electron-oxidized form) is required but is not available for the "
                    f"'{state_id}' state. Attach precomputed energy data or calculate it from "
                    "scratch (see the attach_energy() and calculate_electronic_structure() "
                    "methods)"
                )
                break

        return _errmsg

    def _calculate_global_descriptors_redox(self) -> Optional[str]:
        """Calculate the global C-DFT descriptors and store them in the global feature cache.

        Returns
        -------
        Optional[str]
            An error message if the calculation of the global descriptors failed, otherwise
            ``None``.
        """
        # Check if the global descriptors have already been calculated
        if "global-ionization_potential" in self.global_feature_cache[self.conformer_idx]:
            return None

        # Calculate all data
        _energy_n = cast(Tuple[float, str], self.energy_n)  # for type checker
        _energy_n_minus1 = cast(Tuple[float, str], self.energy_n_minus1)  # for type checker
        _energy_n_plus1 = cast(Tuple[float, str], self.energy_n_plus1)  # for type checker
        (
            error_message,
            ionization_potential,
            electron_affinity,
            chem_potential,
            hardness,
            softness,
            electrophilicity,
            nucleophilicity,
        ) = calculate_global_cdft_descriptors_redox(
            energy_n=_energy_n,
            energy_n_minus1=_energy_n_minus1,
            energy_n_plus1=_energy_n_plus1,
        )
        if error_message is not None:
            return error_message

        # Write the data to the global feature cache
        self.global_feature_cache[self.conformer_idx]["global-ionization_potential"] = (
            ionization_potential
        )
        self.global_feature_cache[self.conformer_idx]["global-electron_affinity"] = (
            electron_affinity
        )
        self.global_feature_cache[self.conformer_idx]["global-chem_potential_redox"] = (
            chem_potential
        )
        self.global_feature_cache[self.conformer_idx]["global-hardness_redox"] = hardness
        self.global_feature_cache[self.conformer_idx]["global-softness_redox"] = softness
        self.global_feature_cache[self.conformer_idx]["global-electrophilicity_redox"] = (
            electrophilicity
        )
        self.global_feature_cache[self.conformer_idx]["global-nucleophilicity_redox"] = (
            nucleophilicity
        )

        return None
