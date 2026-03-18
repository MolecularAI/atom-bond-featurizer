"""Atom and orbital population features from ``Multiwfn``."""

from bonafide.features.multiwfn_population_analysis import _Multiwfn3DAtomPopulationAnalysis
from bonafide.utils.typing_protocols import _MultiwfnMixinProtocol


class _LowdinAlphaBetaMixin:
    """Mixin class for the Lowdin alpha and beta atom population features."""

    def calculate(self: _MultiwfnMixinProtocol) -> None:
        """Calculate the features."""
        if self.multiplicity == 1:
            self.results[0] = {self.feature_name: "_inaccessible"}
            return

        self._run_multiwfn(command_list=[6, "\n"])
        self._read_output_file3(scheme_name="lowdin")


class _MullikenAlphaBetaMixin:
    """Mixin class for the Mulliken alpha and beta atom population features."""

    def calculate(self: _MultiwfnMixinProtocol) -> None:
        """Calculate the features."""
        if self.multiplicity == 1:
            self.results[0] = {self.feature_name: "_inaccessible"}
            return

        self._run_multiwfn(command_list=[5, 1])
        self._read_output_file3(scheme_name="mulliken")


class _MullikenBickelhauptAlphaBetaMixin:
    """Mixin class for the Mulliken-Bickelhaupt alpha and beta atom population features."""

    def calculate(self: _MultiwfnMixinProtocol) -> None:
        """Calculate the features."""
        if self.multiplicity == 1:
            self.results[0] = {self.feature_name: "_inaccessible"}
            return

        self._run_multiwfn(command_list=[9])
        self._read_output_file3(scheme_name="mulliken_bickelhaupt")


class _MullikenRosSchuitAlphaBetaMixin:
    """Mixin class for the Mulliken-Ros-Schuit alpha and beta atom population features."""

    def calculate(self: _MultiwfnMixinProtocol) -> None:
        """Calculate the features."""
        if self.multiplicity == 1:
            self.results[0] = {self.feature_name: "_inaccessible"}
            return

        self._run_multiwfn(command_list=[7])
        self._read_output_file3(scheme_name="mulliken_ros_schuit")


class _MullikenStoutPolitzerAlphaBetaMixin:
    """Mixin class for the Mulliken-Stout-Politzer alpha and beta atom population features."""

    def calculate(self: _MultiwfnMixinProtocol) -> None:
        """Calculate the features."""
        if self.multiplicity == 1:
            self.results[0] = {self.feature_name: "_inaccessible"}
            return

        self._run_multiwfn(command_list=[8])
        self._read_output_file3(scheme_name="mulliken_stout_politzer")


class Multiwfn3DAtomPopulationLowdin(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_lowdin", calculated with multiwfn.

    The index of this feature is 300 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_lowdin`` feature."""
        self._run_multiwfn(command_list=[6, "\n"])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="lowdin")
        else:
            self._read_output_file3(scheme_name="lowdin")


class Multiwfn3DAtomPopulationLowdinAlpha(_Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin):
    """Feature factory for the 3D atom feature "population_lowdin_alpha", calculated with
    multiwfn.

    The index of this feature is 301 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinBeta(_Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin):
    """Feature factory for the 3D atom feature "population_lowdin_beta", calculated with
    multiwfn.

    The index of this feature is 302 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinD(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_lowdin_d", calculated with multiwfn.

    The index of this feature is 303 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_lowdin_d`` feature."""
        self._run_multiwfn(command_list=[6, "\n"])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="lowdin")
        else:
            self._read_output_file3(scheme_name="lowdin")


class Multiwfn3DAtomPopulationLowdinDAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_lowdin_d_alpha", calculated with
    multiwfn.

    The index of this feature is 304 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinDBeta(_Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin):
    """Feature factory for the 3D atom feature "population_lowdin_d_beta", calculated with
    multiwfn.

    The index of this feature is 305 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinF(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_lowdin_f", calculated with multiwfn.

    The index of this feature is 306 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_lowdin_f`` feature."""
        self._run_multiwfn(command_list=[6, "\n"])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="lowdin")
        else:
            self._read_output_file3(scheme_name="lowdin")


class Multiwfn3DAtomPopulationLowdinFAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_lowdin_f_alpha", calculated with
    multiwfn.

    The index of this feature is 307 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinFBeta(_Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin):
    """Feature factory for the 3D atom feature "population_lowdin_f_beta", calculated with
    multiwfn.

    The index of this feature is 308 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinG(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_lowdin_g", calculated with multiwfn.

    The index of this feature is 309 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_lowdin_g`` feature."""
        self._run_multiwfn(command_list=[6, "\n"])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="lowdin")
        else:
            self._read_output_file3(scheme_name="lowdin")


class Multiwfn3DAtomPopulationLowdinGAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_lowdin_g_alpha", calculated with
    multiwfn.

    The index of this feature is 310 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinGBeta(_Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin):
    """Feature factory for the 3D atom feature "population_lowdin_g_beta", calculated with
    multiwfn.

    The index of this feature is 311 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinH(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_lowdin_h", calculated with multiwfn.

    The index of this feature is 312 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_lowdin_h`` feature."""
        self._run_multiwfn(command_list=[6, "\n"])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="lowdin")
        else:
            self._read_output_file3(scheme_name="lowdin")


class Multiwfn3DAtomPopulationLowdinHAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_lowdin_h_alpha", calculated with
    multiwfn.

    The index of this feature is 313 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinHBeta(_Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin):
    """Feature factory for the 3D atom feature "population_lowdin_h_beta", calculated with
    multiwfn.

    The index of this feature is 314 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinP(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_lowdin_p", calculated with multiwfn.

    The index of this feature is 315 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_lowdin_p`` feature."""
        self._run_multiwfn(command_list=[6, "\n"])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="lowdin")
        else:
            self._read_output_file3(scheme_name="lowdin")


class Multiwfn3DAtomPopulationLowdinPAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_lowdin_p_alpha", calculated with
    multiwfn.

    The index of this feature is 316 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinPBeta(_Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin):
    """Feature factory for the 3D atom feature "population_lowdin_p_beta", calculated with
    multiwfn.

    The index of this feature is 317 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinS(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_lowdin_s", calculated with multiwfn.

    The index of this feature is 318 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_lowdin_s`` feature."""
        self._run_multiwfn(command_list=[6, "\n"])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="lowdin")
        else:
            self._read_output_file3(scheme_name="lowdin")


class Multiwfn3DAtomPopulationLowdinSAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_lowdin_s_alpha", calculated with
    multiwfn.

    The index of this feature is 319 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationLowdinSBeta(_Multiwfn3DAtomPopulationAnalysis, _LowdinAlphaBetaMixin):
    """Feature factory for the 3D atom feature "population_lowdin_s_beta", calculated with
    multiwfn.

    The index of this feature is 320 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _LowdinAlphaBetaMixin


class Multiwfn3DAtomPopulationMulliken(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_mulliken", calculated with multiwfn.

    The index of this feature is 321 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_mulliken`` feature."""
        self._run_multiwfn(command_list=[5, 1])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken")
        else:
            self._read_output_file3(scheme_name="mulliken")


class Multiwfn3DAtomPopulationMullikenAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_alpha", calculated with
    multiwfn.

    The index of this feature is 322 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenBeta(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_beta", calculated with
    multiwfn.

    The index of this feature is 323 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenBickelhaupt(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_mulliken_bickelhaupt", calculated
    with multiwfn.

    The index of this feature is 324 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_mulliken_bickelhaupt`` feature."""
        self._run_multiwfn(command_list=[9])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken_bickelhaupt")
        else:
            self._read_output_file3(scheme_name="mulliken_bickelhaupt")


class Multiwfn3DAtomPopulationMullikenBickelhauptAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenBickelhauptAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_bickelhaupt_alpha",
    calculated with multiwfn.

    The index of this feature is 325 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenBickelhauptAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenBickelhauptBeta(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenBickelhauptAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_bickelhaupt_beta",
    calculated with multiwfn.

    The index of this feature is 326 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenBickelhauptAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenD(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_mulliken_d", calculated with
    multiwfn.

    The index of this feature is 327 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_mulliken_d`` feature."""
        self._run_multiwfn(command_list=[5, 1])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken")
        else:
            self._read_output_file3(scheme_name="mulliken")


class Multiwfn3DAtomPopulationMullikenDAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_d_alpha", calculated with
    multiwfn.

    The index of this feature is 328 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenDBeta(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_d_beta", calculated with
    multiwfn.

    The index of this feature is 329 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenF(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_mulliken_f", calculated with
    multiwfn.

    The index of this feature is 330 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_mulliken_f`` feature."""
        self._run_multiwfn(command_list=[5, 1])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken")
        else:
            self._read_output_file3(scheme_name="mulliken")


class Multiwfn3DAtomPopulationMullikenFAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_f_alpha", calculated with
    multiwfn.

    The index of this feature is 331 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenFBeta(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_f_beta", calculated with
    multiwfn.

    The index of this feature is 332 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenG(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_mulliken_g", calculated with
    multiwfn.

    The index of this feature is 333 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_mulliken_g`` feature."""
        self._run_multiwfn(command_list=[5, 1])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken")
        else:
            self._read_output_file3(scheme_name="mulliken")


class Multiwfn3DAtomPopulationMullikenGAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_g_alpha", calculated with
    multiwfn.

    The index of this feature is 334 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenGBeta(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_g_beta", calculated with
    multiwfn.

    The index of this feature is 335 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenH(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_mulliken_h", calculated with
    multiwfn.

    The index of this feature is 336 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_mulliken_h`` feature."""
        self._run_multiwfn(command_list=[5, 1])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken")
        else:
            self._read_output_file3(scheme_name="mulliken")


class Multiwfn3DAtomPopulationMullikenHAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_h_alpha", calculated with
    multiwfn.

    The index of this feature is 337 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenHBeta(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_h_beta", calculated with
    multiwfn.

    The index of this feature is 338 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenP(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_mulliken_p", calculated with
    multiwfn.

    The index of this feature is 339 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_mulliken_p`` feature."""
        self._run_multiwfn(command_list=[5, 1])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken")
        else:
            self._read_output_file3(scheme_name="mulliken")


class Multiwfn3DAtomPopulationMullikenPAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_p_alpha", calculated with
    multiwfn.

    The index of this feature is 340 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenPBeta(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_p_beta", calculated with
    multiwfn.

    The index of this feature is 341 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenRosSchuit(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_mulliken_ros_schuit", calculated with
    multiwfn.

    The index of this feature is 342 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_mulliken_ros_schuit`` feature."""
        self._run_multiwfn(command_list=[7])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken_ros_schuit")
        else:
            self._read_output_file3(scheme_name="mulliken_ros_schuit")


class Multiwfn3DAtomPopulationMullikenRosSchuitAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenRosSchuitAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_ros_schuit_alpha",
    calculated with multiwfn.

    The index of this feature is 343 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenRosSchuitAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenRosSchuitBeta(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenRosSchuitAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_ros_schuit_beta", calculated
    with multiwfn.

    The index of this feature is 344 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenRosSchuitAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenS(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_mulliken_s", calculated with
    multiwfn.

    The index of this feature is 345 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_mulliken_s`` feature."""
        self._run_multiwfn(command_list=[5, 1])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken")
        else:
            self._read_output_file3(scheme_name="mulliken")


class Multiwfn3DAtomPopulationMullikenSAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_s_alpha", calculated with
    multiwfn.

    The index of this feature is 346 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenSBeta(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_s_beta", calculated with
    multiwfn.

    The index of this feature is 347 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenStoutPolitzer(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "population_mulliken_stout_politzer", calculated
    with multiwfn.

    The index of this feature is 348 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-population_mulliken_stout_politzer`` feature."""
        self._run_multiwfn(command_list=[8])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken_stout_politzer")
        else:
            self._read_output_file3(scheme_name="mulliken_stout_politzer")


class Multiwfn3DAtomPopulationMullikenStoutPolitzerAlpha(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenStoutPolitzerAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_stout_politzer_alpha",
    calculated with multiwfn.

    The index of this feature is 349 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenStoutPolitzerAlphaBetaMixin


class Multiwfn3DAtomPopulationMullikenStoutPolitzerBeta(
    _Multiwfn3DAtomPopulationAnalysis, _MullikenStoutPolitzerAlphaBetaMixin
):
    """Feature factory for the 3D atom feature "population_mulliken_stout_politzer_beta",
    calculated with multiwfn.

    The index of this feature is 350 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature is calculated in _MullikenStoutPolitzerAlphaBetaMixin
