"""Atom and bond features from ``qmdesc``."""

from qmdesc import ReactivityDescriptorHandler

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.helper_functions_chemistry import get_atom_bond_mapping_dicts


class _Qmdesc2DAtom(BaseFeaturizer):
    """Parent feature factory for the 2D atom qmdesc features."""

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the qmdesc atom features."""
        self._run_qmdesc()

    def _run_qmdesc(self) -> None:
        """Run qmdesc and write the atom features to the ``results`` dictionary.

        Returns
        -------
        None
        """
        mapping_dict_atoms, _, canonical_smiles = get_atom_bond_mapping_dicts(self.mol)
        handler = ReactivityDescriptorHandler()
        predictions = handler.predict(canonical_smiles)

        for idx, value in enumerate(predictions["fukui_elec"]):
            if idx in mapping_dict_atoms:
                self.results[mapping_dict_atoms[idx]] = {"qmdesc2D-atom-fukui_plus": float(value)}

        for idx, value in enumerate(predictions["fukui_neu"]):
            if idx in mapping_dict_atoms:
                self.results[mapping_dict_atoms[idx]]["qmdesc2D-atom-fukui_minus"] = float(value)

        for idx, value in enumerate(predictions["partial_charge"]):
            if idx in mapping_dict_atoms:
                self.results[mapping_dict_atoms[idx]]["qmdesc2D-atom-partial_charge"] = float(value)

        for idx, value in enumerate(predictions["NMR"]):
            if idx in mapping_dict_atoms:
                self.results[mapping_dict_atoms[idx]]["qmdesc2D-atom-nmr_chemical_shift"] = float(
                    value
                )

        for idx, (value_a, value_b) in enumerate(
            zip(predictions["fukui_elec"], predictions["fukui_neu"])
        ):
            if idx in mapping_dict_atoms:
                self.results[mapping_dict_atoms[idx]]["qmdesc2D-atom-fukui_dual"] = float(
                    value_a
                ) - float(value_b)


class Qmdesc2DAtomFukuiDual(_Qmdesc2DAtom):
    """Feature factory for the 2D atom feature "fukui_dual", calculated with qmdesc.

    The index of this feature is 484 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "qmdesc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Qmdesc2DAtom


class Qmdesc2DAtomFukuiMinus(_Qmdesc2DAtom):
    """Feature factory for the 2D atom feature "fukui_minus", calculated with qmdesc.

    The index of this feature is 485 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "qmdesc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Qmdesc2DAtom


class Qmdesc2DAtomFukuiPlus(_Qmdesc2DAtom):
    """Feature factory for the 2D atom feature "fukui_plus", calculated with qmdesc.

    The index of this feature is 486 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "qmdesc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Qmdesc2DAtom


class Qmdesc2DAtomNmrChemicalShift(_Qmdesc2DAtom):
    """Feature factory for the 2D atom feature "nmr_chemical_shift", calculated with qmdesc.

    The index of this feature is 487 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "qmdesc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Qmdesc2DAtom


class Qmdesc2DAtomPartialCharge(_Qmdesc2DAtom):
    """Feature factory for the 2D atom feature "partial_charge", calculated with qmdesc.

    The index of this feature is 488 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "qmdesc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Qmdesc2DAtom


class _Qmdesc2DBond(BaseFeaturizer):
    """Parent feature factory for qmdesc's bond features."""

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the qmdesc bond features."""
        self._run_qmdesc()

    def _run_qmdesc(self) -> None:
        """Run qmdesc and write the bond features to the ``results`` dictionary.

        Returns
        -------
        None
        """
        _, mapping_dict_bonds, canonical_smiles = get_atom_bond_mapping_dicts(self.mol)
        handler = ReactivityDescriptorHandler()
        predictions = handler.predict(canonical_smiles)

        for idx, value in enumerate(predictions["bond_order"]):
            if idx in mapping_dict_bonds:
                self.results[mapping_dict_bonds[idx]] = {"qmdesc2D-bond-bond_order": float(value)}

        for idx, value in enumerate(predictions["bond_length"]):
            if idx in mapping_dict_bonds:
                self.results[mapping_dict_bonds[idx]]["qmdesc2D-bond-bond_length"] = float(value)


class Qmdesc2DBondBondLength(_Qmdesc2DBond):
    """Feature factory for the 2D bond feature "bond_length", calculated with qmdesc.

    The index of this feature is 489 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "qmdesc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Qmdesc2DBond


class Qmdesc2DBondBondOrder(_Qmdesc2DBond):
    """Feature factory for the 2D bond feature "bond_order", calculated with qmdesc.

    The index of this feature is 490 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "qmdesc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Qmdesc2DBond
