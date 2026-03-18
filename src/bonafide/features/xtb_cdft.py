"""C-DFT features based on condensed Fukui indices calculated with xtb."""

from typing import Optional, Tuple

from bonafide.features.xtb_fukui_misc import (
    Xtb3DAtomCdftCondensedFukuiDual,
    Xtb3DAtomCdftCondensedFukuiMinus,
    Xtb3DAtomCdftCondensedFukuiPlus,
    Xtb3DAtomCdftCondensedFukuiZero,
)
from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.cdft_redox_mixin import CdftLocalRedoxMixin


class _Xtb3DAtomCdftLocal(BaseFeaturizer, CdftLocalRedoxMixin):
    """Parent feature factory for calculating C-DFT descriptors based on condensed Fukui indices
    calculated with xtb.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _wrap_fukui(
        self, factory: type, fukui_feature_name: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Calculate the condensed Fukui indices with xtb.

        The calculation is performed with the feature factory class of the respective Fukui
        coefficient.

        Parameters
        ----------
        factory : type
            The feature factory class to be used for the Fukui calculation.
        fukui_feature_name : str
            The name of the Fukui feature to be calculated.

        Returns
        -------
        Tuple[Optional[float], Optional[str]]
            The calculated Fukui value and an error message.
        """
        fukui = None
        error_message = None

        # Save the initial feature name
        _feature_name = self.feature_name

        # Initialize the calculation of the Fukui values. In case they were already calculated
        # in a previous feature calculation, they are automatically fetched from the cache.
        calc = factory()

        # Temporarily set the feature name to calculate the Fukui value
        params = self.__dict__
        params["feature_name"] = fukui_feature_name

        # Get the Fukui value
        fukui, error_message = calc(**params)

        # Reset the feature name back to the actual feature name
        self.feature_name = _feature_name

        return fukui, error_message


class Xtb3DAtomCdftLocalElectrophilicityFmo(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_electrophilicity_fmo", calculated
    with xtb.

    The index of this feature is 566 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_electrophilicity_fmo`` feature."""
        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiPlus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_plus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_electrophilicity = self.global_feature_cache[self.conformer_idx][
            "xtb3D-global-electrophilicity_fmo"
        ]

        assert isinstance(global_electrophilicity, (int, float))  # for type checker
        assert isinstance(fukui_plus, (int, float))  # for type checker

        local_electrophilicity = round(number=global_electrophilicity * fukui_plus, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_electrophilicity


class Xtb3DAtomCdftLocalElectrophilicityRedox(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_electrophilicity_redox", calculated
    with xtb.

    The index of this feature is 567 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_electrophilicity_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiPlus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_plus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on ionization potential and electron affinity and
        # write them to the global feature cache
        error_message = self._calculate_global_descriptors_redox()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_electrophilicity = self.global_feature_cache[self.conformer_idx][
            "global-electrophilicity_redox"
        ]

        assert isinstance(global_electrophilicity, (int, float))  # for type checker
        assert isinstance(fukui_plus, (int, float))  # for type checker

        local_electrophilicity = round(number=global_electrophilicity * fukui_plus, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_electrophilicity


class Xtb3DAtomCdftLocalHardnessMinusFmo(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hardness_minus_fmo", calculated with
    xtb.

    The index of this feature is 568 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_hardness_minus_fmo`` feature."""
        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiMinus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_minus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_hardness = self.global_feature_cache[self.conformer_idx]["xtb3D-global-hardness_fmo"]

        assert isinstance(global_hardness, (int, float))  # for type checker
        assert isinstance(fukui_minus, (int, float))  # for type checker

        local_hardness_minus = round(number=global_hardness * fukui_minus, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_hardness_minus


class Xtb3DAtomCdftLocalHardnessMinusRedox(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hardness_minus_redox", calculated
    with xtb.

    The index of this feature is 569 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_hardness_minus_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiMinus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_minus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on ionization potential and electron affinity and
        # write them to the global feature cache
        error_message = self._calculate_global_descriptors_redox()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_hardness = self.global_feature_cache[self.conformer_idx]["global-hardness_redox"]

        assert isinstance(global_hardness, (int, float))  # for type checker
        assert isinstance(fukui_minus, (int, float))  # for type checker

        local_hardness_minus = round(number=global_hardness * fukui_minus, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_hardness_minus


class Xtb3DAtomCdftLocalHardnessPlusFmo(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hardness_plus_fmo", calculated with
    xtb.

    The index of this feature is 570 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_hardness_plus_fmo`` feature."""
        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiPlus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_plus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_hardness = self.global_feature_cache[self.conformer_idx]["xtb3D-global-hardness_fmo"]

        assert isinstance(global_hardness, (int, float))  # for type checker
        assert isinstance(fukui_plus, (int, float))  # for type checker

        local_hardness_plus = round(number=global_hardness * fukui_plus, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_hardness_plus


class Xtb3DAtomCdftLocalHardnessPlusRedox(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hardness_plus_redox", calculated with
    xtb.

    The index of this feature is 571 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_hardness_plus_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiPlus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_plus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on ionization potential and electron affinity and
        # write them to the global feature cache
        error_message = self._calculate_global_descriptors_redox()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_hardness = self.global_feature_cache[self.conformer_idx]["global-hardness_redox"]

        assert isinstance(global_hardness, (int, float))  # for type checker
        assert isinstance(fukui_plus, (int, float))  # for type checker

        local_hardness_plus = round(number=global_hardness * fukui_plus, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_hardness_plus


class Xtb3DAtomCdftLocalHardnessZeroFmo(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hardness_zero_fmo", calculated with
    xtb.

    The index of this feature is 572 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_hardness_zero_fmo`` feature."""
        # Calculate the Fukui zero value which is required for the calculation of this feature
        fukui_zero, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiZero,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_zero",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_hardness = self.global_feature_cache[self.conformer_idx]["xtb3D-global-hardness_fmo"]

        assert isinstance(global_hardness, (int, float))  # for type checker
        assert isinstance(fukui_zero, (int, float))  # for type checker

        local_hardness_zero = round(number=global_hardness * fukui_zero, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_hardness_zero


class Xtb3DAtomCdftLocalHardnessZeroRedox(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hardness_zero_redox", calculated with
    xtb.

    The index of this feature is 573 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_hardness_zero_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui zero value which is required for the calculation of this feature
        fukui_zero, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiZero,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_zero",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on ionization potential and electron affinity and
        # write them to the global feature cache
        error_message = self._calculate_global_descriptors_redox()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_hardness = self.global_feature_cache[self.conformer_idx]["global-hardness_redox"]

        assert isinstance(global_hardness, (int, float))  # for type checker
        assert isinstance(fukui_zero, (int, float))  # for type checker

        local_hardness_zero = round(number=global_hardness * fukui_zero, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_hardness_zero


class Xtb3DAtomCdftLocalHyperhardnessFmo(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hyperhardness_fmo", calculated with
    xtb.

    The index of this feature is 574 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_hyperhardness_fmo`` feature."""
        # Calculate the dual descriptor which is required for the calculation of this feature
        fukui_dual, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiDual,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_dual",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_hardness = self.global_feature_cache[self.conformer_idx]["xtb3D-global-hardness_fmo"]

        assert isinstance(global_hardness, (int, float))  # for type checker
        assert isinstance(fukui_dual, (int, float))  # for type checker

        local_hyperhardness = round(number=global_hardness**2 * fukui_dual, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_hyperhardness


class Xtb3DAtomCdftLocalHyperhardnessRedox(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hyperhardness_redox", calculated with
    xtb.

    The index of this feature is 575 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_hyperhardness_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the dual descriptor which is required for the calculation of this feature
        fukui_dual, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiDual,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_dual",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on ionization potential and electron affinity and
        # write them to the global feature cache
        error_message = self._calculate_global_descriptors_redox()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_hardness = self.global_feature_cache[self.conformer_idx]["global-hardness_redox"]

        assert isinstance(global_hardness, (int, float))  # for type checker
        assert isinstance(fukui_dual, (int, float))  # for type checker

        local_hyperhardness = round(number=global_hardness**2 * fukui_dual, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_hyperhardness


class Xtb3DAtomCdftLocalHypersoftnessFmo(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hypersoftness_fmo", calculated with
    xtb.

    The index of this feature is 576 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_hypersoftness_fmo`` feature."""
        # Calculate the dual descriptor which is required for the calculation of this feature
        fukui_dual, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiDual,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_dual",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_softness = self.global_feature_cache[self.conformer_idx]["xtb3D-global-softness_fmo"]

        assert isinstance(global_softness, (int, float))  # for type checker
        assert isinstance(fukui_dual, (int, float))  # for type checker

        local_hypersoftness = round(number=global_softness**2 * fukui_dual, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_hypersoftness


class Xtb3DAtomCdftLocalHypersoftnessRedox(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hypersoftness_redox", calculated with
    xtb.

    The index of this feature is 577 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_hypersoftness_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the dual descriptor which is required for the calculation of this feature
        fukui_dual, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiDual,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_dual",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on ionization potential and electron affinity and
        # write them to the global feature cache
        error_message = self._calculate_global_descriptors_redox()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_softness = self.global_feature_cache[self.conformer_idx]["global-softness_redox"]

        assert isinstance(global_softness, (int, float))  # for type checker
        assert isinstance(fukui_dual, (int, float))  # for type checker

        local_hypersoftness = round(number=global_softness**2 * fukui_dual, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_hypersoftness


class Xtb3DAtomCdftLocalNucleophilicityFmo(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_nucleophilicity_fmo", calculated with
    xtb.

    The index of this feature is 578 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_nucleophilicity_fmo`` feature."""
        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiMinus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_minus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_nucleophilicity = self.global_feature_cache[self.conformer_idx][
            "xtb3D-global-nucleophilicity_fmo"
        ]

        assert isinstance(global_nucleophilicity, (int, float))  # for type checker
        assert isinstance(fukui_minus, (int, float))  # for type checker

        local_nucleophilicity = round(number=global_nucleophilicity * fukui_minus, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_nucleophilicity


class Xtb3DAtomCdftLocalNucleophilicityRedox(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_nucleophilicity_redox", calculated
    with xtb.

    The index of this feature is 579 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_nucleophilicity_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiMinus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_minus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on ionization potential and electron affinity and
        # write them to the global feature cache
        error_message = self._calculate_global_descriptors_redox()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_nucleophilicity = self.global_feature_cache[self.conformer_idx][
            "global-nucleophilicity_redox"
        ]

        assert isinstance(global_nucleophilicity, (int, float))  # for type checker
        assert isinstance(fukui_minus, (int, float))  # for type checker

        local_nucleophilicity = round(number=global_nucleophilicity * fukui_minus, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_nucleophilicity


class Xtb3DAtomCdftLocalRelativeElectrophilicity(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_relative_electrophilicity",
    calculated with xtb.

    The index of this feature is 580 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_relative_electrophilicity`` feature."""
        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiMinus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_minus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiPlus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_plus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        assert isinstance(fukui_minus, (int, float))  # for type checker
        assert isinstance(fukui_plus, (int, float))  # for type checker

        try:
            relative_electrophilicity = round(number=fukui_plus / fukui_minus, ndigits=6)
        except ZeroDivisionError:
            self._err = (
                "cannot be calculated because the Fukui minus coefficient (denominator) is zero"
            )
            return

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = relative_electrophilicity


class Xtb3DAtomCdftLocalRelativeNucleophilicity(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_relative_nucleophilicity", calculated
    with xtb.

    The index of this feature is 581 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_relative_nucleophilicity`` feature."""
        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiMinus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_minus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiPlus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_plus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        assert isinstance(fukui_minus, (int, float))  # for type checker
        assert isinstance(fukui_plus, (int, float))  # for type checker

        try:
            relative_nucleophilicity = round(number=fukui_minus / fukui_plus, ndigits=6)
        except ZeroDivisionError:
            self._err = (
                "cannot be calculated because the Fukui plus coefficient (denominator) is zero"
            )
            return

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = relative_nucleophilicity


class Xtb3DAtomCdftLocalSoftnessMinusFmo(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_softness_minus_fmo", calculated with
    xtb.

    The index of this feature is 582 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_softness_minus_fmo`` feature."""
        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiMinus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_minus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_softness = self.global_feature_cache[self.conformer_idx]["xtb3D-global-softness_fmo"]

        assert isinstance(global_softness, (int, float))  # for type checker
        assert isinstance(fukui_minus, (int, float))  # for type checker

        local_softness_minus = round(number=global_softness * fukui_minus, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_softness_minus


class Xtb3DAtomCdftLocalSoftnessMinusRedox(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_softness_minus_redox", calculated
    with xtb.

    The index of this feature is 583 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_softness_minus_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiMinus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_minus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on ionization potential and electron affinity and
        # write them to the global feature cache
        error_message = self._calculate_global_descriptors_redox()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_softness = self.global_feature_cache[self.conformer_idx]["global-softness_redox"]

        assert isinstance(global_softness, (int, float))  # for type checker
        assert isinstance(fukui_minus, (int, float))  # for type checker

        local_softness_minus = round(number=global_softness * fukui_minus, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_softness_minus


class Xtb3DAtomCdftLocalSoftnessPlusFmo(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_softness_plus_fmo", calculated with
    xtb.

    The index of this feature is 584 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_softness_plus_fmo`` feature."""
        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiPlus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_plus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_softness = self.global_feature_cache[self.conformer_idx]["xtb3D-global-softness_fmo"]

        assert isinstance(global_softness, (int, float))  # for type checker
        assert isinstance(fukui_plus, (int, float))  # for type checker

        local_softness_plus = round(number=global_softness * fukui_plus, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_softness_plus


class Xtb3DAtomCdftLocalSoftnessPlusRedox(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_softness_plus_redox", calculated with
    xtb.

    The index of this feature is 585 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_softness_plus_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiPlus,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_plus",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on ionization potential and electron affinity and
        # write them to the global feature cache
        error_message = self._calculate_global_descriptors_redox()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_softness = self.global_feature_cache[self.conformer_idx]["global-softness_redox"]

        assert isinstance(global_softness, (int, float))  # for type checker
        assert isinstance(fukui_plus, (int, float))  # for type checker

        local_softness_plus = round(number=global_softness * fukui_plus, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_softness_plus


class Xtb3DAtomCdftLocalSoftnessZeroFmo(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_softness_zero_fmo", calculated with
    xtb.

    The index of this feature is 586 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_softness_zero_fmo`` feature."""
        # Calculate the Fukui zero value which is required for the calculation of this feature
        fukui_zero, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiZero,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_zero",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_softness = self.global_feature_cache[self.conformer_idx]["xtb3D-global-softness_fmo"]

        assert isinstance(global_softness, (int, float))  # for type checker
        assert isinstance(fukui_zero, (int, float))  # for type checker

        local_softness_zero = round(number=global_softness * fukui_zero, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_softness_zero


class Xtb3DAtomCdftLocalSoftnessZeroRedox(_Xtb3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_softness_zero_redox", calculated with
    xtb.

    The index of this feature is 587 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "xtb" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``xtb3D-atom-cdft_local_softness_zero_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui zero value which is required for the calculation of this feature
        fukui_zero, error_message = self._wrap_fukui(
            factory=Xtb3DAtomCdftCondensedFukuiZero,
            fukui_feature_name="xtb3D-atom-cdft_condensed_fukui_zero",
        )
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on ionization potential and electron affinity and
        # write them to the global feature cache
        error_message = self._calculate_global_descriptors_redox()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_softness = self.global_feature_cache[self.conformer_idx]["global-softness_redox"]

        assert isinstance(global_softness, (int, float))  # for type checker
        assert isinstance(fukui_zero, (int, float))  # for type checker

        local_softness_zero = round(number=global_softness * fukui_zero, ndigits=6)

        if self.atom_bond_idx not in self.results:
            self.results[self.atom_bond_idx] = {}
        self.results[self.atom_bond_idx][self.feature_name] = local_softness_zero
