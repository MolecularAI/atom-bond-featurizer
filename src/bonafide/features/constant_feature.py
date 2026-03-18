"""Constant values for atoms or bonds as a feature."""

from bonafide.utils.base_featurizer import BaseFeaturizer


class Bonafide2DAtomConstantFeature(BaseFeaturizer):
    """Feature factory for the 2D atom feature "constant_feature", implemented within this
    package.

    The index of this feature is 17 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.constant" in the _feature_config.toml file.
    """

    atom_constant: str

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-constant_feature`` feature."""
        self.results[self.atom_bond_idx] = {self.feature_name: self.atom_constant}


class Bonafide2DBondConstantFeature(BaseFeaturizer):
    """Feature factory for the 2D bond feature "constant_feature", implemented within this
    package.

    The index of this feature is 42 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.constant" in the _feature_config.toml file.
    """

    bond_constant: str

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-constant_feature`` feature."""
        self.results[self.atom_bond_idx] = {self.feature_name: self.bond_constant}
