"""Partial charge features from ``Multiwfn``."""

from typing import List, Union

from bonafide.features.multiwfn_population_analysis import _Multiwfn3DAtomPopulationAnalysis
from bonafide.utils.io_ import write_sd_file


class Multiwfn3DAtomPartialChargeBecke(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_becke", calculated with multiwfn.

    The index of this feature is 273 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    n_iterations_becke_partition: int
    radius_becke_partition: int

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_becke`` feature."""
        # Select the Becke partitioning method
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [10]

        # Select atomic radii definition
        multiwfn_commands.extend([1, self.radius_becke_partition])

        # Select number of iterations
        multiwfn_commands.extend([2, self.n_iterations_becke_partition])

        # Run calculation
        multiwfn_commands.append(0)

        self._run_multiwfn(command_list=multiwfn_commands)
        self._read_output_file(feature_name="multiwfn3D-atom-partial_charge_becke")


class Multiwfn3DAtomPartialChargeChelpg(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_chelpg", calculated with
    multiwfn.

    The index of this feature is 283 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    atomic_radii: int
    box_extension_chelpg: float
    esp_type: int
    exclude_atoms: List[int]
    grid_spacing_chelpg: float

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_chelpg`` feature."""
        # Select CHELPG method
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [12]

        # Set grid spacing
        multiwfn_commands.extend([2, self.grid_spacing_chelpg])

        # Set box extension
        multiwfn_commands.extend([3, self.box_extension_chelpg])

        # Exclude atoms (if requested)
        if self.exclude_atoms != []:
            multiwfn_commands.extend(
                [
                    4,
                    ",".join(
                        [
                            str(atom.GetIdx() + 1)
                            for atom in self.mol.GetAtoms()
                            if atom.GetIdx() not in self.exclude_atoms
                        ]
                    ),
                ]
            )

        # Chose ESP type
        multiwfn_commands.extend([5, self.esp_type])

        # Choose atomic radius definition
        multiwfn_commands.extend([10, self.atomic_radii])

        # Run calculation
        multiwfn_commands.append(1)
        self._run_multiwfn(command_list=multiwfn_commands, from_resp_chelpg=True)
        self._read_output_file4(feature_name="multiwfn3D-atom-partial_charge_chelpg")


class Multiwfn3DAtomPartialChargeCm5(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_cm5", calculated with multiwfn.

    The index of this feature is 284 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_cm5`` feature."""
        self._run_multiwfn(command_list=[16, 1])
        self._read_output_file(feature_name="multiwfn3D-atom-partial_charge_cm5")


class Multiwfn3DAtomPartialChargeCm5Scaled1point2(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_cm5_scaled_1point2", calculated
    with multiwfn.

    The index of this feature is 285 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_cm5_scaled_1point2`` feature."""
        self._run_multiwfn(command_list=[-16, 1])
        self._read_output_file(feature_name="multiwfn3D-atom-partial_charge_cm5_scaled_1point2")


class Multiwfn3DAtomPartialChargeCorrectedHirshfeld(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_corrected_hirshfeld", calculated
    with multiwfn.

    The index of this feature is 286 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_corrected_hirshfeld`` feature."""
        self._run_multiwfn(command_list=[11, 1])
        self._read_output_file(feature_name="multiwfn3D-atom-partial_charge_corrected_hirshfeld")


class Multiwfn3DAtomPartialChargeEem(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_eem", calculated with multiwfn.

    The index of this feature is 287 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    eem_parameters: int

    electronic_struc_n: str

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_eem`` feature."""
        # Generate the required SD file for calculating the EEM charges and write it to the
        # electronic_struc_n attribute for input to Multiwfn
        write_sd_file(mol=self.mol, file_path=f"{self.conformer_name}.sdf")

        _electronic_struc_n = self.electronic_struc_n
        self.electronic_struc_n = f"{self.conformer_name}.sdf"

        # Select EEM method
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [17]

        # Choose EEM parameters
        multiwfn_commands.extend([1, self.eem_parameters, -2])

        # Set charge
        multiwfn_commands.extend([2, str(self.charge)])

        # Run calculation
        multiwfn_commands.append(0)
        self._run_multiwfn(command_list=multiwfn_commands, from_eem=True)

        # Reset self.electronic_struc_n (just to be safe)
        self.electronic_struc_n = _electronic_struc_n

        self._read_output_file5(feature_name="multiwfn3D-atom-partial_charge_eem")


class Multiwfn3DAtomPartialChargeHirshfeld(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_hirshfeld", calculated with
    multiwfn.

    The index of this feature is 288 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_hirshfeld`` feature."""
        self._run_multiwfn(command_list=[1, 1])
        self._read_output_file(feature_name="multiwfn3D-atom-partial_charge_hirshfeld")


class Multiwfn3DAtomPartialChargeLowdin(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_lowdin", calculated with
    multiwfn.

    The index of this feature is 289 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_lowdin`` feature."""
        self._run_multiwfn(command_list=[6, "\n"])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="lowdin")
        else:
            self._read_output_file3(scheme_name="lowdin")


class Multiwfn3DAtomPartialChargeMerzKollmann(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_merz_kollmann", calculated with
    multiwfn.

    The index of this feature is 290 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    atomic_radii: int
    esp_type: int
    exclude_atoms: List[int]
    fitting_points_settings_merz_kollmann: List[float]
    n_points_angstrom2_merz_kollmann: float

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_merz_kollmann`` feature."""
        # Select calculation of MK charges
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [13]

        # Set number of points per Angstrom^2
        multiwfn_commands.extend([2, self.n_points_angstrom2_merz_kollmann])

        # Set number and scale factors of the layers of the MK fitting points
        multiwfn_commands.append(3)
        for scale_factor in self.fitting_points_settings_merz_kollmann:
            multiwfn_commands.append(scale_factor)
        multiwfn_commands.append("q")

        # Exclude atoms (if requested)
        if self.exclude_atoms != []:
            multiwfn_commands.extend(
                [
                    4,
                    ",".join(
                        [
                            str(atom.GetIdx() + 1)
                            for atom in self.mol.GetAtoms()
                            if atom.GetIdx() not in self.exclude_atoms
                        ]
                    ),
                ]
            )

        # Chose ESP type
        multiwfn_commands.extend([5, self.esp_type])

        # Choose atomic radius definition
        multiwfn_commands.extend([10, self.atomic_radii])

        # Run calculation
        multiwfn_commands.append(1)
        self._run_multiwfn(command_list=multiwfn_commands, from_resp_chelpg=True)
        self._read_output_file4(feature_name="multiwfn3D-atom-partial_charge_merz_kollmann")


class Multiwfn3DAtomPartialChargeMulliken(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_mulliken", calculated with
    multiwfn.

    The index of this feature is 291 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_mulliken`` feature."""
        self._run_multiwfn(command_list=[5, 1])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken")
        else:
            self._read_output_file3(scheme_name="mulliken")


class Multiwfn3DAtomPartialChargeMullikenBickelhaupt(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_mulliken_bickelhaupt", calculated
    with multiwfn.

    The index of this feature is 292 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_mulliken_bickelhaupt`` feature."""
        self._run_multiwfn(command_list=[9])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken_bickelhaupt")
        else:
            self._read_output_file3(scheme_name="mulliken_bickelhaupt")


class Multiwfn3DAtomPartialChargeMullikenRosSchuit(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_mulliken_ros_schuit", calculated
    with multiwfn.

    The index of this feature is 293 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_mulliken_ros_schuit`` feature."""
        self._run_multiwfn(command_list=[7])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken_ros_schuit")
        else:
            self._read_output_file3(scheme_name="mulliken_ros_schuit")


class Multiwfn3DAtomPartialChargeMullikenStoutPolitzer(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_mulliken_stout_politzer",
    calculated with multiwfn.

    The index of this feature is 294 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_mulliken_stout_politzer`` feature."""
        self._run_multiwfn(command_list=[8])
        if self.multiplicity == 1:
            self._read_output_file2(scheme_name="mulliken_stout_politzer")
        else:
            self._read_output_file3(scheme_name="mulliken_stout_politzer")


class Multiwfn3DAtomPartialChargeRespChelpgOneStage(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_resp_chelpg_one_stage",
    calculated with multiwfn.

    The index of this feature is 295 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    atomic_radii: int
    box_extension_chelpg: float
    ch_equivalence_constraint_resp: bool
    convergence_threshold_resp: float
    esp_type: int
    grid_spacing_chelpg: float
    n_iterations_resp: int
    restraint_one_stage_resp: float
    restraint_stage1_resp: float
    restraint_stage2_resp: float
    tightness_resp: float

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_resp_chelpg_one_stage`` feature."""
        # Select RESP method
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [18]

        # Set method and parameters for distributing fitting points to CHELPG
        multiwfn_commands.extend([3, 2])

        # Set CHELPG grid spacing
        multiwfn_commands.extend([1, self.grid_spacing_chelpg])

        # Set CHELPG box extension
        multiwfn_commands.extend([2, self.box_extension_chelpg])

        # Go back to main RESP menu
        multiwfn_commands.append(0)

        # Set hyperbolic penalty and various other running parameters
        multiwfn_commands.append(4)
        multiwfn_commands.extend([1, self.tightness_resp])
        multiwfn_commands.extend([2, self.restraint_one_stage_resp])
        multiwfn_commands.extend([3, self.restraint_stage1_resp])
        multiwfn_commands.extend([4, self.restraint_stage2_resp])
        multiwfn_commands.extend([5, self.n_iterations_resp])
        multiwfn_commands.extend([6, self.convergence_threshold_resp])
        multiwfn_commands.append(0)

        # Set equivalence constraint in fitting
        if self.ch_equivalence_constraint_resp is False:
            multiwfn_commands.extend([5, 0])
        else:
            multiwfn_commands.extend([5, 2])

        # Choose atomic radius definition
        multiwfn_commands.extend([10, self.atomic_radii])

        # Chose ESP type
        multiwfn_commands.extend([11, self.esp_type])

        # Start one-stage RESP fitting
        multiwfn_commands.append(2)
        self._run_multiwfn(command_list=multiwfn_commands, from_resp_chelpg=True)
        self._read_output_file4(feature_name="multiwfn3D-atom-partial_charge_resp_chelpg_one_stage")


class Multiwfn3DAtomPartialChargeRespChelpgTwoStage(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_resp_chelpg_two_stage",
    calculated with multiwfn.

    The index of this feature is 296 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    atomic_radii: int
    box_extension_chelpg: float
    ch_equivalence_constraint_resp: bool
    convergence_threshold_resp: float
    esp_type: int
    grid_spacing_chelpg: float
    n_iterations_resp: int
    restraint_one_stage_resp: float
    restraint_stage1_resp: float
    restraint_stage2_resp: float
    tightness_resp: float

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_resp_chelpg_two_stage`` feature."""
        # Select RESP method
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [18]

        # Set method and parameters for distributing fitting points to CHELPG
        multiwfn_commands.extend([3, 2])

        # Set CHELPG grid spacing
        multiwfn_commands.extend([1, self.grid_spacing_chelpg])

        # Set CHELPG box extension
        multiwfn_commands.extend([2, self.box_extension_chelpg])

        # Go back to main RESP menu
        multiwfn_commands.append(0)

        # Set hyperbolic penalty and various other running parameters
        multiwfn_commands.append(4)
        multiwfn_commands.extend([1, self.tightness_resp])
        multiwfn_commands.extend([2, self.restraint_one_stage_resp])
        multiwfn_commands.extend([3, self.restraint_stage1_resp])
        multiwfn_commands.extend([4, self.restraint_stage2_resp])
        multiwfn_commands.extend([5, self.n_iterations_resp])
        multiwfn_commands.extend([6, self.convergence_threshold_resp])
        multiwfn_commands.append(0)

        # Set equivalence constraint in fitting
        if self.ch_equivalence_constraint_resp is False:
            multiwfn_commands.extend([5, 0])
        else:
            multiwfn_commands.extend([5, 2])

        # Choose atomic radius definition
        multiwfn_commands.extend([10, self.atomic_radii])

        # Chose ESP type
        multiwfn_commands.extend([11, self.esp_type])

        # Start two-stage RESP fitting
        multiwfn_commands.append(1)
        self._run_multiwfn(command_list=multiwfn_commands, from_resp_chelpg=True)
        self._read_output_file4(feature_name="multiwfn3D-atom-partial_charge_resp_chelpg_two_stage")


class Multiwfn3DAtomPartialChargeRespMerzKollmannOneStage(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_resp_merz_kollmann_one_stage",
    calculated with multiwfn.

    The index of this feature is 297 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    atomic_radii: int
    ch_equivalence_constraint_resp: bool
    convergence_threshold_resp: float
    esp_type: int
    fitting_points_settings_merz_kollmann: List[float]
    n_iterations_resp: int
    n_points_angstrom2_merz_kollmann: float
    restraint_one_stage_resp: float
    restraint_stage1_resp: float
    restraint_stage2_resp: float
    tightness_resp: float

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_resp_merz_kollmann_one_stage``
        feature."""
        # Select RESP method
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [18]

        # Set method and parameters for distributing fitting points to Merz-Kollmann
        multiwfn_commands.extend([3, 1])

        # Set number of points per Angstrom^2
        multiwfn_commands.extend([1, self.n_points_angstrom2_merz_kollmann])

        # Set number of layers per atom
        multiwfn_commands.extend([2, len(self.fitting_points_settings_merz_kollmann)])

        # Set the value times van der Waals radius in each layer
        multiwfn_commands.append(3)
        for scale_factor in self.fitting_points_settings_merz_kollmann:
            multiwfn_commands.append(scale_factor)

        # Go back to main RESP menu
        multiwfn_commands.append(0)

        # Set hyperbolic penalty and various other running parameters
        multiwfn_commands.append(4)
        multiwfn_commands.extend([1, self.tightness_resp])
        multiwfn_commands.extend([2, self.restraint_one_stage_resp])
        multiwfn_commands.extend([3, self.restraint_stage1_resp])
        multiwfn_commands.extend([4, self.restraint_stage2_resp])
        multiwfn_commands.extend([5, self.n_iterations_resp])
        multiwfn_commands.extend([6, self.convergence_threshold_resp])
        multiwfn_commands.append(0)

        # Set equivalence constraint in fitting
        if self.ch_equivalence_constraint_resp is False:
            multiwfn_commands.extend([5, 0])
        else:
            multiwfn_commands.extend([5, 2])

        # Choose atomic radius definition
        multiwfn_commands.extend([10, self.atomic_radii])

        # Chose ESP type
        multiwfn_commands.extend([11, self.esp_type])

        # Start one-stage RESP fitting
        multiwfn_commands.append(2)
        self._run_multiwfn(command_list=multiwfn_commands, from_resp_chelpg=True)
        self._read_output_file4(
            feature_name="multiwfn3D-atom-partial_charge_resp_merz_kollmann_one_stage"
        )


class Multiwfn3DAtomPartialChargeRespMerzKollmannTwoStage(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_resp_merz_kollmann_two_stage",
    calculated with multiwfn.

    The index of this feature is 298 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    atomic_radii: int
    ch_equivalence_constraint_resp: bool
    convergence_threshold_resp: float
    esp_type: int
    fitting_points_settings_merz_kollmann: List[float]
    n_iterations_resp: int
    n_points_angstrom2_merz_kollmann: float
    restraint_one_stage_resp: float
    restraint_stage1_resp: float
    restraint_stage2_resp: float
    tightness_resp: float

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_resp_merz_kollmann_two_stage``
        feature."""
        # Select RESP method
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [18]

        # Set method and parameters for distributing fitting points to Merz-Kollmann
        multiwfn_commands.extend([3, 1])

        # Set number of points per Angstrom^2
        multiwfn_commands.extend([1, self.n_points_angstrom2_merz_kollmann])

        # Set number of layers per atom
        multiwfn_commands.extend([2, len(self.fitting_points_settings_merz_kollmann)])

        # Set the value times van der Waals radius in each layer
        multiwfn_commands.append(3)
        for scale_factor in self.fitting_points_settings_merz_kollmann:
            multiwfn_commands.append(scale_factor)

        # Go back to main RESP menu
        multiwfn_commands.append(0)

        # Set hyperbolic penalty and various other running parameters
        multiwfn_commands.append(4)
        multiwfn_commands.extend([1, self.tightness_resp])
        multiwfn_commands.extend([2, self.restraint_one_stage_resp])
        multiwfn_commands.extend([3, self.restraint_stage1_resp])
        multiwfn_commands.extend([4, self.restraint_stage2_resp])
        multiwfn_commands.extend([5, self.n_iterations_resp])
        multiwfn_commands.extend([6, self.convergence_threshold_resp])
        multiwfn_commands.append(0)

        # Set equivalence constraint in fitting
        if self.ch_equivalence_constraint_resp is False:
            multiwfn_commands.extend([5, 0])
        else:
            multiwfn_commands.extend([5, 2])

        # Choose atomic radius definition
        multiwfn_commands.extend([10, self.atomic_radii])

        # Chose ESP type
        multiwfn_commands.extend([11, self.esp_type])

        # Start two-stage RESP fitting
        multiwfn_commands.append(1)
        self._run_multiwfn(command_list=multiwfn_commands, from_resp_chelpg=True)
        self._read_output_file4(
            feature_name="multiwfn3D-atom-partial_charge_resp_merz_kollmann_two_stage"
        )


class Multiwfn3DAtomPartialChargeVdd(_Multiwfn3DAtomPopulationAnalysis):
    """Feature factory for the 3D atom feature "partial_charge_vdd", calculated with multiwfn.

    The index of this feature is 299 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.population" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-partial_charge_vdd`` feature."""
        self._run_multiwfn(command_list=[2, 1])
        self._read_output_file(feature_name="multiwfn3D-atom-partial_charge_vdd")
