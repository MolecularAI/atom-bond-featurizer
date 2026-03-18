"""Atom and orbital spin population features from ``Multiwfn``."""

from bonafide.features.multiwfn_population_analysis import _Multiwfn3DAtomPopulationAnalysis
from bonafide.utils.typing_protocols import _MultiwfnMixinProtocol


class _LowdinMixin:
    """Mixin class for the Lowdin atom spin population features."""

    def calculate(self: _MultiwfnMixinProtocol) -> None:
        """Calculate the features."""
        if self.multiplicity == 1:
            for atom in self.mol.GetAtoms():
                self.results[atom.GetIdx()] = {self.feature_name: 0.0}
            return

        self._run_multiwfn(command_list=[6, "\n"])
        self._read_output_file3(scheme_name="lowdin")


class _MullikenMixin:
    """Mixin class for the Mulliken atom spin population features."""

    def calculate(self: _MultiwfnMixinProtocol) -> None:
        """Calculate the features."""
        if self.multiplicity == 1:
            for atom in self.mol.GetAtoms():
                self.results[atom.GetIdx()] = {self.feature_name: 0.0}
            return

        self._run_multiwfn(command_list=[5, 1])
        self._read_output_file3(scheme_name="mulliken")


class Multiwfn3DAtomSpinPopulationLowdin(_Multiwfn3DAtomPopulationAnalysis, _LowdinMixin):
    """Feature factory for the 3D atom feature "spin_population_lowdin", calculated with
    multiwfn.

    The index of this feature is 351 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinMixin


class Multiwfn3DAtomSpinPopulationLowdinD(_Multiwfn3DAtomPopulationAnalysis, _LowdinMixin):
    """Feature factory for the 3D atom feature "spin_population_lowdin_d", calculated with
    multiwfn.

    The index of this feature is 352 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinMixin


class Multiwfn3DAtomSpinPopulationLowdinF(_Multiwfn3DAtomPopulationAnalysis, _LowdinMixin):
    """Feature factory for the 3D atom feature "spin_population_lowdin_f", calculated with
    multiwfn.

    The index of this feature is 353 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinMixin


class Multiwfn3DAtomSpinPopulationLowdinG(_Multiwfn3DAtomPopulationAnalysis, _LowdinMixin):
    """Feature factory for the 3D atom feature "spin_population_lowdin_g", calculated with
    multiwfn.

    The index of this feature is 354 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinMixin


class Multiwfn3DAtomSpinPopulationLowdinH(_Multiwfn3DAtomPopulationAnalysis, _LowdinMixin):
    """Feature factory for the 3D atom feature "spin_population_lowdin_h", calculated with
    multiwfn.

    The index of this feature is 355 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinMixin


class Multiwfn3DAtomSpinPopulationLowdinP(_Multiwfn3DAtomPopulationAnalysis, _LowdinMixin):
    """Feature factory for the 3D atom feature "spin_population_lowdin_p", calculated with
    multiwfn.

    The index of this feature is 356 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinMixin


class Multiwfn3DAtomSpinPopulationLowdinS(_Multiwfn3DAtomPopulationAnalysis, _LowdinMixin):
    """Feature factory for the 3D atom feature "spin_population_lowdin_s", calculated with
    multiwfn.

    The index of this feature is 357 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinMixin


class Multiwfn3DAtomSpinPopulationMulliken(_Multiwfn3DAtomPopulationAnalysis, _MullikenMixin):
    """Feature factory for the 3D atom feature "spin_population_mulliken", calculated with
    multiwfn.

    The index of this feature is 358 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenMixin


class Multiwfn3DAtomSpinPopulationMullikenBickelhaupt(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "spin_population_mulliken_bickelhaupt",
    calculated with multiwfn.

    The index of this feature is 359 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-spin_population_mulliken_bickelhaupt`` feature."""
        if self.multiplicity == 1:
            for atom in self.mol.GetAtoms():
                self.results[atom.GetIdx()] = {self.feature_name: 0.0}
            return

        self._run_multiwfn(command_list=[9])
        self._read_output_file3(scheme_name="mulliken_bickelhaupt")


class Multiwfn3DAtomSpinPopulationMullikenD(_Multiwfn3DAtomPopulationAnalysis, _MullikenMixin):
    """Feature factory for the 3D atom feature "spin_population_mulliken_d", calculated with
    multiwfn.

    The index of this feature is 360 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenMixin


class Multiwfn3DAtomSpinPopulationMullikenF(_Multiwfn3DAtomPopulationAnalysis, _MullikenMixin):
    """Feature factory for the 3D atom feature "spin_population_mulliken_f", calculated with
    multiwfn.

    The index of this feature is 361 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenMixin


class Multiwfn3DAtomSpinPopulationMullikenG(_Multiwfn3DAtomPopulationAnalysis, _MullikenMixin):
    """Feature factory for the 3D atom feature "spin_population_mulliken_g", calculated with
    multiwfn.

    The index of this feature is 362 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenMixin


class Multiwfn3DAtomSpinPopulationMullikenH(_Multiwfn3DAtomPopulationAnalysis, _MullikenMixin):
    """Feature factory for the 3D atom feature "spin_population_mulliken_h", calculated with
    multiwfn.

    The index of this feature is 363 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenMixin


class Multiwfn3DAtomSpinPopulationMullikenP(_Multiwfn3DAtomPopulationAnalysis, _MullikenMixin):
    """Feature factory for the 3D atom feature "spin_population_mulliken_p", calculated with
    multiwfn.

    The index of this feature is 364 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenMixin


class Multiwfn3DAtomSpinPopulationMullikenRosSchuit(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "spin_population_mulliken_ros_schuit", calculated
    with multiwfn.

    The index of this feature is 365 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-spin_population_mulliken_ros_schuit`` feature."""
        if self.multiplicity == 1:
            for atom in self.mol.GetAtoms():
                self.results[atom.GetIdx()] = {self.feature_name: 0.0}
            return

        self._run_multiwfn(command_list=[7])
        self._read_output_file3(scheme_name="mulliken_ros_schuit")


class Multiwfn3DAtomSpinPopulationMullikenS(_Multiwfn3DAtomPopulationAnalysis, _MullikenMixin):
    """Feature factory for the 3D atom feature "spin_population_mulliken_s", calculated with
    multiwfn.

    The index of this feature is 366 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenMixin


class Multiwfn3DAtomSpinPopulationMullikenStoutPolitzer(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "spin_population_mulliken_stout_politzer",
    calculated with multiwfn.

    The index of this feature is 367 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-spin_population_mulliken_stout_politzer`` feature."""
        if self.multiplicity == 1:
            for atom in self.mol.GetAtoms():
                self.results[atom.GetIdx()] = {self.feature_name: 0.0}
            return

        self._run_multiwfn(command_list=[8])
        self._read_output_file3(scheme_name="mulliken_stout_politzer")
