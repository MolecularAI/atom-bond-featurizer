"""Features from ``mendeleev``."""

import random
from typing import Dict, Union

import numpy as np
from mendeleev import element

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.helper_functions_chemistry import from_periodic_table


class _Mendeleev2DAtom(BaseFeaturizer):
    """Parent feature factory for the 2D atom mendeleev features.

    For details, please refer to the Mendeleev documentation
    (https://mendeleev.readthedocs.io/en/stable/, last accessed on 09.09.2025).
    """

    _periodic_table: Dict[str, element]

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _pull_from_periodic_table(self) -> element:
        """Get the element data from the periodic table.

        Returns
        -------
        element
            The mendeleev element object for the atom of interest.
        """
        _element_symbol = self.mol.GetAtomWithIdx(self.atom_bond_idx).GetSymbol()
        self._periodic_table, element_data = from_periodic_table(
            self._periodic_table, _element_symbol
        )
        return element_data

    def _get_attribute_data(self, none_to_false: bool = False) -> None:
        """Get an atom property as an attribute from the mendeleev element object and write it to
        the results dictionary.

        Parameters
        ----------
        none_to_false : bool, optional
            Whether to convert ``None`` values to ``False``, by default ``False``.

        Returns
        -------
        None
        """
        element_data = self._pull_from_periodic_table()
        _mendeleev_feature_name = self.feature_name.split("-")[-1]
        value = getattr(element_data, _mendeleev_feature_name)

        # Convert numpy float64 to Python float
        if isinstance(value, np.float64):
            value = float(value)

        # Convert None to False if requested (relevant for Mendeleev2DAtomIsMonoisotopic)
        if value is None and none_to_false is True:
            value = False
        elif value is None:
            value = "_inaccessible"

        # Save results in the results dictionary
        self.results[self.atom_bond_idx] = {self.feature_name: value}

    def _get_en_data(self, **kwargs: Union[str, int]) -> None:
        """Get the electronegativity data from the mendeleev element object and write it to the
        results dictionary.

        Parameters
        ----------
        **kwargs: Union[str, int]
            Optional keyword arguments to pass to the ``electronegativity()`` method of the
            mendeleev ``element`` object.

        Returns
        -------
        None
        """
        if kwargs.get("set_as_inaccessible", False) is True:
            self.results[self.atom_bond_idx] = {self.feature_name: "_inaccessible"}
            return

        element_data = self._pull_from_periodic_table()
        value = element_data.electronegativity(**kwargs)
        if isinstance(value, np.float64):
            value = float(value)
        self.results[self.atom_bond_idx] = {self.feature_name: value}


class Mendeleev2DAtomAtomicRadius(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "atomic_radius", calculated with mendeleev.

    The index of this feature is 103 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-atomic_radius`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomAtomicRadiusRahm(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "atomic_radius_rahm", calculated with mendeleev.

    The index of this feature is 104 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-atomic_radius_rahm`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomAtomicVolume(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "atomic_volume", calculated with mendeleev.

    The index of this feature is 105 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-atomic_volume`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomAtomicWeight(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "atomic_weight", calculated with mendeleev.

    The index of this feature is 106 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-atomic_weight`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomBlock(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "block", calculated with mendeleev.

    The index of this feature is 107 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-block`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomC6(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "c6", calculated with mendeleev.

    The index of this feature is 108 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-c6`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomC6Gb(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "c6_gb", calculated with mendeleev.

    The index of this feature is 109 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-c6_gb`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomCas(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "cas", calculated with mendeleev.

    The index of this feature is 110 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-cas`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomCovalentRadiusBragg(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "covalent_radius_bragg", calculated with
    mendeleev.

    The index of this feature is 111 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-covalent_radius_bragg`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomCovalentRadiusCordero(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "covalent_radius_cordero", calculated with
    mendeleev.

    The index of this feature is 112 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-covalent_radius_cordero`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomCovalentRadiusPyykko(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "covalent_radius_pyykko", calculated with
    mendeleev.

    The index of this feature is 113 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-covalent_radius_pyykko`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomCovalentRadiusPyykkoDouble(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "covalent_radius_pyykko_double", calculated with
    mendeleev.

    The index of this feature is 114 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-covalent_radius_pyykko_double`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomCovalentRadiusPyykkoTriple(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "covalent_radius_pyykko_triple", calculated with
    mendeleev.

    The index of this feature is 115 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-covalent_radius_pyykko_triple`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomDensity(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "density", calculated with mendeleev.

    The index of this feature is 116 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-density`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomDipolePolarizability(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "dipole_polarizability", calculated with
    mendeleev.

    The index of this feature is 117 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-dipole_polarizability`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomEconf(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "econf", calculated with mendeleev.

    The index of this feature is 118 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-econf`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomElectronAffinity(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "electron_affinity", calculated with mendeleev.

    The index of this feature is 119 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-electron_affinity`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomElectrophilicity(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "electrophilicity", calculated with mendeleev.

    The index of this feature is 120 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-electrophilicity`` feature."""
        element_data = self._pull_from_periodic_table()
        self.results[self.atom_bond_idx] = {self.feature_name: element_data.electrophilicity()}


class Mendeleev2DAtomEnAllen(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "en_allen", calculated with mendeleev.

    The index of this feature is 121 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-en_allen`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomEnAllredRochow(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "en_allred_rochow", calculated with mendeleev.

    The index of this feature is 122 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-en_allred_rochow`` feature."""
        self._get_en_data(scale="allred-rochow")


class Mendeleev2DAtomEnCottrellSutton(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "en_cottrell_sutton", calculated with mendeleev.

    The index of this feature is 123 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-en_cottrell_sutton`` feature."""
        self._get_en_data(scale="cottrell-sutton")


class Mendeleev2DAtomEnGhosh(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "en_ghosh", calculated with mendeleev.

    The index of this feature is 124 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-en_ghosh`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomEnGordy(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "en_gordy", calculated with mendeleev.

    The index of this feature is 125 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-en_gordy`` feature."""
        self._get_en_data(scale="gordy")


class Mendeleev2DAtomEnMartynovBatsanov(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "en_martynov_batsanov", calculated with
    mendeleev.

    The index of this feature is 126 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-en_martynov_batsanov`` feature."""
        self._get_en_data(scale="martynov-batsanov")


class Mendeleev2DAtomEnMiedema(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "en_miedema", calculated with mendeleev.

    The index of this feature is 127 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-en_miedema`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomEnMulliken(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "en_mulliken", calculated with mendeleev.

    The index of this feature is 128 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-en_mulliken`` feature."""
        _formal_charge = self.mol.GetAtomWithIdx(self.atom_bond_idx).GetFormalCharge()

        # Not defined for atoms with formal negative charge
        if _formal_charge < 0:
            self._get_en_data(set_as_inaccessible=True)
            return

        self._get_en_data(scale="mulliken", charge=_formal_charge)


class Mendeleev2DAtomEnNagle(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "en_nagle", calculated with mendeleev.

    The index of this feature is 129 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-en_nagle`` feature."""
        self._get_en_data(scale="nagle")


class Mendeleev2DAtomEnPauling(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "en_pauling", calculated with mendeleev.

    The index of this feature is 130 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-en_pauling`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomEnSanderson(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "en_sanderson", calculated with mendeleev.

    The index of this feature is 131 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-en_sanderson`` feature."""
        self._get_en_data(scale="sanderson")


class Mendeleev2DAtomEvaporationHeat(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "evaporation_heat", calculated with mendeleev.

    The index of this feature is 132 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-evaporation_heat`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomFirstIonizationEnergy(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "first_ionization_energy", calculated with
    mendeleev.

    The index of this feature is 133 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-first_ionization_energy`` feature."""
        element_data = self._pull_from_periodic_table()
        self.results[self.atom_bond_idx] = {self.feature_name: element_data.ionenergies[1]}


class Mendeleev2DAtomFusionHeat(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "fusion_heat", calculated with mendeleev.

    The index of this feature is 134 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-fusion_heat`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomGasBasicity(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "gas_basicity", calculated with mendeleev.

    The index of this feature is 135 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-gas_basicity`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomGroupId(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "group_id", calculated with mendeleev.

    The index of this feature is 136 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-group_id`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomHardness(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "hardness", calculated with mendeleev.

    The index of this feature is 137 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-hardness`` feature."""
        _formal_charge = self.mol.GetAtomWithIdx(self.atom_bond_idx).GetFormalCharge()

        # Not defined for atoms with formal negative charge
        if _formal_charge < 0:
            self.results[self.atom_bond_idx] = {self.feature_name: "_inaccessible"}
            return

        element_data = self._pull_from_periodic_table()
        self.results[self.atom_bond_idx] = {
            self.feature_name: element_data.hardness(charge=_formal_charge)
        }


class Mendeleev2DAtomHeatOfFormation(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "heat_of_formation", calculated with mendeleev.

    The index of this feature is 138 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-heat_of_formation`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomInchi(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "inchi", calculated with mendeleev.

    The index of this feature is 139 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-inchi`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomIsMonoisotopic(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "is_monoisotopic", calculated with mendeleev.

    The index of this feature is 140 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-is_monoisotopic`` feature."""
        self._get_attribute_data(none_to_false=True)


class Mendeleev2DAtomIsRadioactive(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "is_radioactive", calculated with mendeleev.

    The index of this feature is 141 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-is_radioactive`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomLatticeConstant(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "lattice_constant", calculated with mendeleev.

    The index of this feature is 142 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-lattice_constant`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomLatticeStructure(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "lattice_structure", calculated with mendeleev.

    The index of this feature is 143 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-lattice_structure`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomMeltingPoint(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "melting_point", calculated with mendeleev.

    The index of this feature is 144 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-melting_point`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomMetallicRadius(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "metallic_radius", calculated with mendeleev.

    The index of this feature is 145 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-metallic_radius`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomMetallicRadiusC12(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "metallic_radius_c12", calculated with mendeleev.

    The index of this feature is 146 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-metallic_radius_c12`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomMiedemaElectronDensity(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "miedema_electron_density", calculated with
    mendeleev.

    The index of this feature is 147 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-miedema_electron_density`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomMiedemaMolarVolume(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "miedema_molar_volume", calculated with
    mendeleev.

    The index of this feature is 148 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-miedema_molar_volume`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomMolarHeatCapacity(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "molar_heat_capacity", calculated with mendeleev.

    The index of this feature is 149 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-molar_heat_capacity`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomNValenceElectrons(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "n_valence_electrons", calculated with mendeleev.

    The index of this feature is 150 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-n_valence_electrons`` feature."""
        element_data = self._pull_from_periodic_table()
        self.results[self.atom_bond_idx] = {self.feature_name: element_data.nvalence()}


class Mendeleev2DAtomName(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "name", calculated with mendeleev.

    The index of this feature is 151 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-name`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomPeriod(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "period", calculated with mendeleev.

    The index of this feature is 152 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-period`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomPettiforNumber(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "pettifor_number", calculated with mendeleev.

    The index of this feature is 153 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-pettifor_number`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomProtonAffinity(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "proton_affinity", calculated with mendeleev.

    The index of this feature is 154 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-proton_affinity`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomSoftness(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "softness", calculated with mendeleev.

    The index of this feature is 155 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-softness`` feature."""
        _formal_charge = self.mol.GetAtomWithIdx(self.atom_bond_idx).GetFormalCharge()

        # Not defined for atoms with formal negative charge
        if _formal_charge < 0:
            self.results[self.atom_bond_idx] = {self.feature_name: "_inaccessible"}
            return

        element_data = self._pull_from_periodic_table()
        self.results[self.atom_bond_idx] = {
            self.feature_name: element_data.softness(charge=_formal_charge)
        }


class Mendeleev2DAtomSpecificHeatCapacity(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "specific_heat_capacity", calculated with
    mendeleev.

    The index of this feature is 156 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-specific_heat_capacity`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomThermalConductivity(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "thermal_conductivity", calculated with
    mendeleev.

    The index of this feature is 157 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-thermal_conductivity`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomVdwRadius(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "vdw_radius", calculated with mendeleev.

    The index of this feature is 158 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-vdw_radius`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomVdwRadiusAlvarez(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "vdw_radius_alvarez", calculated with mendeleev.

    The index of this feature is 159 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-vdw_radius_alvarez`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomVdwRadiusBatsanov(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "vdw_radius_batsanov", calculated with mendeleev.

    The index of this feature is 160 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-vdw_radius_batsanov`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomVdwRadiusBondi(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "vdw_radius_bondi", calculated with mendeleev.

    The index of this feature is 161 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-vdw_radius_bondi`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomVdwRadiusDreiding(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "vdw_radius_dreiding", calculated with mendeleev.

    The index of this feature is 162 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-vdw_radius_dreiding`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomVdwRadiusMm3(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "vdw_radius_mm3", calculated with mendeleev.

    The index of this feature is 163 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-vdw_radius_mm3`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomVdwRadiusRt(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "vdw_radius_rt", calculated with mendeleev.

    The index of this feature is 164 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-vdw_radius_rt`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomVdwRadiusUff(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "vdw_radius_uff", calculated with mendeleev.

    The index of this feature is 165 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-vdw_radius_uff`` feature."""
        self._get_attribute_data()


class Mendeleev2DAtomZeff(_Mendeleev2DAtom):
    """Feature factory for the 2D atom feature "zeff", calculated with mendeleev.

    The index of this feature is 166 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "mendeleev" in the _feature_config.toml file.
    """

    alle: bool
    method: str

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``mendeleev2D-atom-zeff`` feature."""
        element_data = self._pull_from_periodic_table()
        self.results[self.atom_bond_idx] = {
            self.feature_name: element_data.zeff(method=self.method, alle=self.alle)
        }


# Easter egg
if random.randint(1, 500) == 1:
    print("Today is a good day, and it is time for a smiles.")
