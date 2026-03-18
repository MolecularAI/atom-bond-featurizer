"""Condensed C-DFT features from ``Multiwfn``."""

from typing import Optional, Tuple, Union

from bonafide.features.multiwfn_fukui import (
    Multiwfn3DAtomCdftCondensedFukuiDual,
    Multiwfn3DAtomCdftCondensedFukuiMinus,
    Multiwfn3DAtomCdftCondensedFukuiPlus,
    Multiwfn3DAtomCdftCondensedFukuiZero,
)
from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.cdft_redox_mixin import CdftLocalRedoxMixin
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.global_properties import (
    calculate_global_cdft_descriptors_fmo,
    get_fmo_energies_multiwfn,
)


class _Multiwfn3DAtomCdftLocal(BaseFeaturizer, CdftLocalRedoxMixin):
    """Parent feature factory for the 3D atom C-DFT Multiwfn features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    iterable_option: str

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _check_electronic_structure_data(
        self, el_struc_data: Optional[str], data_name: str
    ) -> Optional[str]:
        """Check if the required electronic structure data is available.

        Parameters
        ----------
        el_struc_data : Optional[str]
            The electronic structure data to check. It is either ``None`` (not available) or the
            path to the electronic structure data file.
        data_name : str
            The identification string of the electronic structure data to check (used for logging).

        Returns
        -------
        Optional[str]
            The error message if the required electronic structure data is not available, ``None``
            if the required electronic structure data is available.
        """
        _errmsg = None
        if el_struc_data is None:
            _errmsg = (
                f"for requesting data from '{self.__class__.__name__}', electronic structure "
                f"data for the {data_name} is required but is not available. Attach precomputed "
                "electronic structure data or calculate it from scratch"
            )
        return _errmsg

    def _calculate_global_descriptors_fmo(self) -> Optional[str]:
        """Calculate molecule-level descriptors from the electronic structure data, that is from
        the frontier molecular orbital energies (HOMO and LUMO).

        The included descriptors are:

        * HOMO energy
        * LUMO energy
        * HOMO-LUMO gap
        * Chemical potential
        * Hardness
        * Softness
        * Electrophilicity
        * Nucleophilicity

        Returns
        -------
        Optional[str]
            The error messages of the subroutines for calculating the global descriptors. ``None``
            if everything worked as expected or if the data is already in the global feature cache.
        """
        # Check if the global descriptors have already been calculated
        if "multiwfn3D-global-homo_energy" in self.global_feature_cache[self.conformer_idx]:
            return None

        # Get relevant environment variables
        environment_variables = {
            var: getattr(self, var, None) for var in PROGRAM_ENVIRONMENT_VARIABLES["multiwfn"]
        }

        # Get the frontier molecular orbital energies
        assert self.electronic_struc_n is not None  # for type checker
        assert self.multiplicity is not None  # for type checker
        homo_energy, lumo_energy, error_message = get_fmo_energies_multiwfn(
            input_file_path=self.electronic_struc_n,
            output_file_name=f"Multiwfn3DFmoEnergies__{self.conformer_name}",
            multiplicity=self.multiplicity,
            environment_variables=environment_variables,
            namespace=self.conformer_name[::-1].split("__", 1)[-1][::-1],
        )
        if error_message is not None:
            return error_message

        # Get the C-DFT descriptors
        assert homo_energy is not None  # for type checker
        assert lumo_energy is not None  # for type checker
        (
            error_message,
            homo_lumo_gap,
            chem_potential,
            hardness,
            softness,
            electrophilicity,
            nucleophilicity,
        ) = calculate_global_cdft_descriptors_fmo(homo_energy=homo_energy, lumo_energy=lumo_energy)
        if error_message is not None:
            return error_message

        # Write the data to the global feature cache
        self.global_feature_cache[self.conformer_idx]["multiwfn3D-global-homo_energy"] = homo_energy
        self.global_feature_cache[self.conformer_idx]["multiwfn3D-global-lumo_energy"] = lumo_energy
        self.global_feature_cache[self.conformer_idx]["multiwfn3D-global-homo_lumo_gap"] = (
            homo_lumo_gap
        )
        self.global_feature_cache[self.conformer_idx]["multiwfn3D-global-chem_potential_fmo"] = (
            chem_potential
        )
        self.global_feature_cache[self.conformer_idx]["multiwfn3D-global-hardness_fmo"] = hardness
        self.global_feature_cache[self.conformer_idx]["multiwfn3D-global-softness_fmo"] = softness
        self.global_feature_cache[self.conformer_idx]["multiwfn3D-global-electrophilicity_fmo"] = (
            electrophilicity
        )
        self.global_feature_cache[self.conformer_idx]["multiwfn3D-global-nucleophilicity_fmo"] = (
            nucleophilicity
        )

        return None

    def _wrap_fukui_dual(self) -> Tuple[Optional[Union[int, float, bool, str]], Optional[str]]:
        """Calculate the condensed dual descriptor with Multiwfn.

        Returns
        -------
        Tuple[Optional[Union[int, float, bool, str]], Optional[str]]
            The condensed dual descriptor and an error message, which is ``None`` if everything
            worked as expected.
        """
        fukui_dual = None
        error_message = None

        # Check if all electronic structure data is available (both redox states needed)
        error_message = self._check_electronic_structure_data(
            el_struc_data=self.electronic_struc_n_plus1,
            data_name="actual molecule plus one electron",
        )
        if error_message is not None:
            return fukui_dual, error_message

        error_message = self._check_electronic_structure_data(
            el_struc_data=self.electronic_struc_n_minus1,
            data_name="actual molecule minus one electron",
        )
        if error_message is not None:
            return fukui_dual, error_message

        # Define actual name of the final feature including the iterable option
        actual_feature_name = f"{self.feature_name}__{self.iterable_option}"

        # Initialize the calculation of the dual descriptor. In case it was already calculated
        # in a previous feature calculation, it is automatically fetched from the cache.
        calc = Multiwfn3DAtomCdftCondensedFukuiDual()

        # Temporarily set the feature name to calculate the Fukui minus value
        params = self.__dict__
        params["feature_name"] = "multiwfn3D-atom-cdft_condensed_fukui_dual"

        # Get the Fukui minus value
        fukui_dual, error_message = calc(**params)

        # Reset the feature name back to the actual feature name
        self.feature_name = actual_feature_name

        return fukui_dual, error_message

    def _wrap_fukui_minus(self) -> Tuple[Optional[Union[int, float, bool, str]], Optional[str]]:
        """Calculate the condensed Fukui minus coefficient with Multiwfn.

        Returns
        -------
        Tuple[Optional[Union[int, float, bool, str]], Optional[str]]
            The condensed Fukui minus coefficient and an error message, which is ``None`` if
            everything worked as expected.
        """
        fukui_minus = None
        error_message = None

        # Check if all electronic structure data is available
        error_message = self._check_electronic_structure_data(
            el_struc_data=self.electronic_struc_n_minus1,
            data_name="actual molecule minus one electron",
        )
        if error_message is not None:
            return fukui_minus, error_message

        # Define actual name of the final feature including the iterable option
        actual_feature_name = f"{self.feature_name}__{self.iterable_option}"

        # Initialize the calculation of the Fukui minus value. In case it was already calculated
        # in a previous feature calculation, it is automatically fetched from the cache.
        calc = Multiwfn3DAtomCdftCondensedFukuiMinus()

        # Temporarily set the feature name to calculate the Fukui minus value
        params = self.__dict__
        params["feature_name"] = "multiwfn3D-atom-cdft_condensed_fukui_minus"

        # Get the Fukui minus value
        fukui_minus, error_message = calc(**params)

        # Reset the feature name back to the actual feature name
        self.feature_name = actual_feature_name

        return fukui_minus, error_message

    def _wrap_fukui_plus(self) -> Tuple[Optional[Union[int, float, bool, str]], Optional[str]]:
        """Calculate the condensed Fukui plus coefficient with Multiwfn.

        Returns
        -------
        Tuple[Optional[Union[int, float, bool, str]], Optional[str]]
            The condensed Fukui plus coefficient and an error message, which is ``None`` if
            everything worked as expected.
        """
        fukui_plus = None
        error_message = None

        # Check if all electronic structure data is available
        error_message = self._check_electronic_structure_data(
            el_struc_data=self.electronic_struc_n_plus1,
            data_name="actual molecule plus one electron",
        )
        if error_message is not None:
            return fukui_plus, error_message

        # Define actual name of the final feature including the iterable option
        actual_feature_name = f"{self.feature_name}__{self.iterable_option}"

        # Initialize the calculation of the Fukui plus value. In case it was already calculated
        # in a previous feature calculation, it is automatically fetched from the cache.
        calc = Multiwfn3DAtomCdftCondensedFukuiPlus()

        # Temporarily set the feature name to calculate the Fukui plus value
        params = self.__dict__
        params["feature_name"] = "multiwfn3D-atom-cdft_condensed_fukui_plus"

        # Get the Fukui plus value
        fukui_plus, error_message = calc(**params)

        # Reset the feature name back to the actual feature name
        self.feature_name = actual_feature_name

        return fukui_plus, error_message

    def _wrap_fukui_zero(self) -> Tuple[Optional[Union[int, float, bool, str]], Optional[str]]:
        """Calculate the condensed Fukui zero coefficient with Multiwfn.

        Returns
        -------
        Tuple[Optional[Union[int, float, bool, str]], Optional[str]]
            The condensed Fukui zero coefficient and an error message, which is ``None`` if
            everything worked as expected.
        """
        fukui_zero = None
        error_message = None

        # Check if all electronic structure data is available (both redox states needed)
        error_message = self._check_electronic_structure_data(
            el_struc_data=self.electronic_struc_n_plus1,
            data_name="actual molecule plus one electron",
        )
        if error_message is not None:
            return fukui_zero, error_message

        error_message = self._check_electronic_structure_data(
            el_struc_data=self.electronic_struc_n_minus1,
            data_name="actual molecule minus one electron",
        )
        if error_message is not None:
            return fukui_zero, error_message

        # Define actual name of the final feature including the iterable option
        actual_feature_name = f"{self.feature_name}__{self.iterable_option}"

        # Initialize the calculation of the Fukui zero value. In case it was already calculated
        # in a previous feature calculation, it is automatically fetched from the cache.
        calc = Multiwfn3DAtomCdftCondensedFukuiZero()

        # Temporarily set the feature name to calculate the Fukui zero value
        params = self.__dict__
        params["feature_name"] = "multiwfn3D-atom-cdft_condensed_fukui_zero"

        # Get the Fukui zero value
        fukui_zero, error_message = calc(**params)

        # Reset the feature name back to the actual feature name
        self.feature_name = actual_feature_name

        return fukui_zero, error_message


class Multiwfn3DAtomCdftLocalElectrophilicityFmo(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_electrophilicity_fmo", calculated
    with multiwfn.

    The index of this feature is 208 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_electrophilicity_fmo`` feature."""
        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui_plus()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on frontier molecular orbitals and write them to the
        # global feature cache
        error_message = self._calculate_global_descriptors_fmo()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_electrophilicity = self.global_feature_cache[self.conformer_idx][
            "multiwfn3D-global-electrophilicity_fmo"
        ]

        assert isinstance(global_electrophilicity, (int, float))  # for type checker
        assert isinstance(fukui_plus, (int, float))  # for type checker

        local_electrophilicity = round(number=global_electrophilicity * fukui_plus, ndigits=6)
        self.results[self.atom_bond_idx][self.feature_name] = local_electrophilicity


class Multiwfn3DAtomCdftLocalElectrophilicityRedox(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_electrophilicity_redox", calculated
    with multiwfn.

    The index of this feature is 209 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_electrophilicity_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui_plus()
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
        self.results[self.atom_bond_idx][self.feature_name] = local_electrophilicity


class Multiwfn3DAtomCdftLocalHardnessMinusFmo(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hardness_minus_fmo", calculated with
    multiwfn.

    The index of this feature is 210 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_hardness_minus_fmo`` feature."""
        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui_minus()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on frontier molecular orbitals and write them to the
        # global feature cache
        error_message = self._calculate_global_descriptors_fmo()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_hardness = self.global_feature_cache[self.conformer_idx][
            "multiwfn3D-global-hardness_fmo"
        ]

        assert isinstance(global_hardness, (int, float))  # for type checker
        assert isinstance(fukui_minus, (int, float))  # for type checker

        local_hardness_minus = round(number=global_hardness * fukui_minus, ndigits=6)
        self.results[self.atom_bond_idx][self.feature_name] = local_hardness_minus


class Multiwfn3DAtomCdftLocalHardnessMinusRedox(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hardness_minus_redox", calculated
    with multiwfn.

    The index of this feature is 211 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_hardness_minus_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui_minus()
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
        self.results[self.atom_bond_idx][self.feature_name] = local_hardness_minus


class Multiwfn3DAtomCdftLocalHardnessPlusFmo(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hardness_plus_fmo", calculated with
    multiwfn.

    The index of this feature is 212 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_hardness_plus_fmo`` feature."""
        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui_plus()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on frontier molecular orbitals and write them to the
        # global feature cache
        error_message = self._calculate_global_descriptors_fmo()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_hardness = self.global_feature_cache[self.conformer_idx][
            "multiwfn3D-global-hardness_fmo"
        ]

        assert isinstance(global_hardness, (int, float))  # for type checker
        assert isinstance(fukui_plus, (int, float))  # for type checker

        local_hardness_plus = round(number=global_hardness * fukui_plus, ndigits=6)
        self.results[self.atom_bond_idx][self.feature_name] = local_hardness_plus


class Multiwfn3DAtomCdftLocalHardnessPlusRedox(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hardness_plus_redox", calculated with
    multiwfn.

    The index of this feature is 213 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_hardness_plus_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui_plus()
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
        self.results[self.atom_bond_idx][self.feature_name] = local_hardness_plus


class Multiwfn3DAtomCdftLocalHardnessZeroFmo(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hardness_zero_fmo", calculated with
    multiwfn.

    The index of this feature is 214 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_hardness_zero_fmo`` feature."""
        # Calculate the Fukui zero value which is required for the calculation of this feature
        fukui_zero, error_message = self._wrap_fukui_zero()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on frontier molecular orbitals and write them to the
        # global feature cache
        error_message = self._calculate_global_descriptors_fmo()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_hardness = self.global_feature_cache[self.conformer_idx][
            "multiwfn3D-global-hardness_fmo"
        ]

        assert isinstance(global_hardness, (int, float))  # for type checker
        assert isinstance(fukui_zero, (int, float))  # for type checker

        local_hardness_zero = round(number=global_hardness * fukui_zero, ndigits=6)
        self.results[self.atom_bond_idx][self.feature_name] = local_hardness_zero


class Multiwfn3DAtomCdftLocalHardnessZeroRedox(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hardness_zero_redox", calculated with
    multiwfn.

    The index of this feature is 215 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_hardness_zero_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui zero value which is required for the calculation of this feature
        fukui_zero, error_message = self._wrap_fukui_zero()
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
        self.results[self.atom_bond_idx][self.feature_name] = local_hardness_zero


class Multiwfn3DAtomCdftLocalHyperhardnessFmo(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hyperhardness_fmo", calculated with
    multiwfn.

    The index of this feature is 216 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_hyperhardness_fmo`` feature."""
        # Calculate the dual descriptor which is required for the calculation of this feature
        fukui_dual, error_message = self._wrap_fukui_dual()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on frontier molecular orbitals and write them to the
        # global feature cache
        error_message = self._calculate_global_descriptors_fmo()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_hardness = self.global_feature_cache[self.conformer_idx][
            "multiwfn3D-global-hardness_fmo"
        ]

        assert isinstance(global_hardness, (int, float))  # for type checker
        assert isinstance(fukui_dual, (int, float))  # for type checker

        local_hyperhardness = round(number=global_hardness**2 * fukui_dual, ndigits=6)
        self.results[self.atom_bond_idx][self.feature_name] = local_hyperhardness


class Multiwfn3DAtomCdftLocalHyperhardnessRedox(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hyperhardness_redox", calculated with
    multiwfn.

    The index of this feature is 217 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_hyperhardness_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the dual descriptor which is required for the calculation of this feature
        fukui_dual, error_message = self._wrap_fukui_dual()
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
        self.results[self.atom_bond_idx][self.feature_name] = local_hyperhardness


class Multiwfn3DAtomCdftLocalHypersoftnessFmo(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hypersoftness_fmo", calculated with
    multiwfn.

    The index of this feature is 218 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_hypersoftness_fmo`` feature."""
        # Calculate the dual descriptor which is required for the calculation of this feature
        fukui_dual, error_message = self._wrap_fukui_dual()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on frontier molecular orbitals and write them to the
        # global feature cache
        error_message = self._calculate_global_descriptors_fmo()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_softness = self.global_feature_cache[self.conformer_idx][
            "multiwfn3D-global-softness_fmo"
        ]

        assert isinstance(global_softness, (int, float))  # for type checker
        assert isinstance(fukui_dual, (int, float))  # for type checker

        local_hypersoftness = round(number=global_softness**2 * fukui_dual, ndigits=6)
        self.results[self.atom_bond_idx][self.feature_name] = local_hypersoftness


class Multiwfn3DAtomCdftLocalHypersoftnessRedox(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_hypersoftness_redox", calculated with
    multiwfn.

    The index of this feature is 219 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_hypersoftness_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the dual descriptor which is required for the calculation of this feature
        fukui_dual, error_message = self._wrap_fukui_dual()
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
        self.results[self.atom_bond_idx][self.feature_name] = local_hypersoftness


class Multiwfn3DAtomCdftLocalNucleophilicityFmo(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_nucleophilicity_fmo", calculated with
    multiwfn.

    The index of this feature is 220 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_nucleophilicity_fmo`` feature."""
        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui_minus()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on frontier molecular orbitals and write them to the
        # global feature cache
        error_message = self._calculate_global_descriptors_fmo()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_nucleophilicity = self.global_feature_cache[self.conformer_idx][
            "multiwfn3D-global-nucleophilicity_fmo"
        ]

        assert isinstance(global_nucleophilicity, (int, float))  # for type checker
        assert isinstance(fukui_minus, (int, float))  # for type checker

        local_nucleophilicity = round(number=global_nucleophilicity * fukui_minus, ndigits=6)
        self.results[self.atom_bond_idx][self.feature_name] = local_nucleophilicity


class Multiwfn3DAtomCdftLocalNucleophilicityRedox(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_nucleophilicity_redox", calculated
    with multiwfn.

    The index of this feature is 221 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_nucleophilicity_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui_minus()
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
        self.results[self.atom_bond_idx][self.feature_name] = local_nucleophilicity


class Multiwfn3DAtomCdftLocalRelativeElectrophilicity(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_relative_electrophilicity",
    calculated with multiwfn.

    The index of this feature is 222 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_relative_electrophilicity`` feature."""
        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui_minus()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui_plus()
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

        self.results[self.atom_bond_idx][self.feature_name] = relative_electrophilicity


class Multiwfn3DAtomCdftLocalRelativeNucleophilicity(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_relative_nucleophilicity", calculated
    with multiwfn.

    The index of this feature is 223 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_relative_nucleophilicity`` feature."""
        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui_minus()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui_plus()
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

        self.results[self.atom_bond_idx][self.feature_name] = relative_nucleophilicity


class Multiwfn3DAtomCdftLocalSoftnessMinusFmo(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_softness_minus_fmo", calculated with
    multiwfn.

    The index of this feature is 224 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_softness_minus_fmo`` feature."""
        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui_minus()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on frontier molecular orbitals and write them to the
        # global feature cache
        error_message = self._calculate_global_descriptors_fmo()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_softness = self.global_feature_cache[self.conformer_idx][
            "multiwfn3D-global-softness_fmo"
        ]

        assert isinstance(global_softness, (int, float))  # for type checker
        assert isinstance(fukui_minus, (int, float))  # for type checker

        local_softness_minus = round(number=global_softness * fukui_minus, ndigits=6)
        self.results[self.atom_bond_idx][self.feature_name] = local_softness_minus


class Multiwfn3DAtomCdftLocalSoftnessMinusRedox(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_softness_minus_redox", calculated
    with multiwfn.

    The index of this feature is 225 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_softness_minus_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui minus value which is required for the calculation of this feature
        fukui_minus, error_message = self._wrap_fukui_minus()
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
        self.results[self.atom_bond_idx][self.feature_name] = local_softness_minus


class Multiwfn3DAtomCdftLocalSoftnessPlusFmo(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_softness_plus_fmo", calculated with
    multiwfn.

    The index of this feature is 226 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_softness_plus_fmo`` feature."""
        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui_plus()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on frontier molecular orbitals and write them to the
        # global feature cache
        error_message = self._calculate_global_descriptors_fmo()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_softness = self.global_feature_cache[self.conformer_idx][
            "multiwfn3D-global-softness_fmo"
        ]

        assert isinstance(global_softness, (int, float))  # for type checker
        assert isinstance(fukui_plus, (int, float))  # for type checker

        local_softness_plus = round(number=global_softness * fukui_plus, ndigits=6)
        self.results[self.atom_bond_idx][self.feature_name] = local_softness_plus


class Multiwfn3DAtomCdftLocalSoftnessPlusRedox(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_softness_plus_redox", calculated with
    multiwfn.

    The index of this feature is 227 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_softness_plus_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui plus value which is required for the calculation of this feature
        fukui_plus, error_message = self._wrap_fukui_plus()
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
        self.results[self.atom_bond_idx][self.feature_name] = local_softness_plus


class Multiwfn3DAtomCdftLocalSoftnessZeroFmo(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_softness_zero_fmo", calculated with
    multiwfn.

    The index of this feature is 228 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_softness_zero_fmo`` feature."""
        # Calculate the Fukui zero value which is required for the calculation of this feature
        fukui_zero, error_message = self._wrap_fukui_zero()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate global descriptors based on frontier molecular orbitals and write them to the
        # global feature cache
        error_message = self._calculate_global_descriptors_fmo()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the desired value and write it to the results dictionary
        global_softness = self.global_feature_cache[self.conformer_idx][
            "multiwfn3D-global-softness_fmo"
        ]

        assert isinstance(global_softness, (int, float))  # for type checker
        assert isinstance(fukui_zero, (int, float))  # for type checker

        local_softness_zero = round(number=global_softness * fukui_zero, ndigits=6)
        self.results[self.atom_bond_idx][self.feature_name] = local_softness_zero


class Multiwfn3DAtomCdftLocalSoftnessZeroRedox(_Multiwfn3DAtomCdftLocal):
    """Feature factory for the 3D atom feature "cdft_local_softness_zero_redox", calculated with
    multiwfn.

    The index of this feature is 229 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.cdft" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-cdft_local_softness_zero_redox`` feature."""
        # Check if energies of all three redox states are available
        error_message = self._check_energy_data()
        if error_message is not None:
            self._err = error_message
            return

        # Calculate the Fukui zero value which is required for the calculation of this feature
        fukui_zero, error_message = self._wrap_fukui_zero()
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
        self.results[self.atom_bond_idx][self.feature_name] = local_softness_zero
