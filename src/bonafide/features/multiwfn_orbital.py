"""Orbital features from ``Multiwfn``."""

import os
from typing import List, Union

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import multiwfn_driver


class _Multiwfn3DAtomMoContribution(BaseFeaturizer):
    """Parent feature factory for the 3D atom orbital Multiwfn features.

    For details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last accessed
    on 12.09.2025).
    """

    homo_minus: int
    lumo_plus: int

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def _run_multiwfn(
        self, mo_type: str, orbital_idx: int, charge_scheme: str, open_shell: Union[bool, str]
    ) -> bool:
        """Run Multiwfn.

        Parameters
        ----------
        mo_type : str
            The type of orbital to consider, either "occupied" or "unoccupied".
        orbital_idx : int
            The index of the orbital defined by the distance (in orbital counts) to the highest
            occupied and lowest unoccupied molecular orbital, respectively (HOMO and LUMO).
        charge_scheme : str
            The name of the population analysis.
        open_shell : Union[bool, str]
            Whether the molecule has an open- or closed-shell electronic structure. This influences
            input formatting and output processing. This is ``False`` for closed-shell cases and
            either ``alpha`` or ``beta`` for open-shell cases.

        Returns
        -------
        bool
            Returns ``False`` if alpha/beta features are requested for closed-shell molecules or
            the total features for open-shell molecules. If the input is correct and Multiwfn is
            executed, ``True`` is returned.
        """
        # Handle differences between open- and closed-shell
        if self.multiplicity == 1 and open_shell in ["alpha", "beta"]:
            # Set one atom feature to _inaccessible, the rest is done by the base featurizer
            self.results[0] = {self.feature_name: "_inaccessible"}
            return False

        if self.multiplicity != 1 and open_shell is False:
            # Set one atom feature to _inaccessible, the rest is done by the base featurizer
            self.results[0] = {self.feature_name: "_inaccessible"}
            return False

        charge_scheme_mapping = {
            "mulliken": 1,
            "mulliken_stout_politzer": 2,
            "mulliken_ros_schuit": 3,
        }

        # Select orbital analysis
        multiwfn_commands: List[Union[str, int, float]]
        multiwfn_commands = [8]

        # Add commands from the child class to the Multiwfn command
        command_list: List[Union[str, int, float]] = [
            charge_scheme_mapping[charge_scheme],
            self._format_orbital_idx(
                mo_type=mo_type, orbital_idx=orbital_idx, open_shell=open_shell
            ),
        ]
        multiwfn_commands.extend(command_list)

        # Exit program
        multiwfn_commands.extend([0, -10, "q"])

        # Set up environment variables
        environment_variables = {
            var: getattr(self, var, None) for var in PROGRAM_ENVIRONMENT_VARIABLES["multiwfn"]
        }

        # Run Multiwfn
        multiwfn_driver(
            cmds=multiwfn_commands,
            input_file_path=str(self.electronic_struc_n),
            output_file_name=f"{self.__class__.__name__}__{self.conformer_name}",
            environment_variables=environment_variables,
            namespace=self.conformer_name[::-1].split("__", 1)[-1][::-1],
        )
        return True

    def _format_orbital_idx(
        self, mo_type: str, orbital_idx: int, open_shell: Union[bool, str]
    ) -> str:
        """Format the orbital index to the format required by Multiwfn.

        Parameters
        ----------
        mo_type : str
            The type of orbital to consider, either "occupied" or "unoccupied".
        orbital_idx : int
            The index of the orbital defined by the distance (in orbital counts) to the highest
            occupied and lowest unoccupied molecular orbital, respectively (HOMO and LUMO).
        open_shell : Union[bool, str]
            Whether the molecule has an open- or closed-shell electronic structure. This influences
            input formatting and output processing. This is ``False`` for closed-shell cases and
            either ``alpha`` or ``beta`` for open-shell cases.

        Returns
        -------
        str
            _description_
        """
        # Handle differences between open- and closed-shell
        insert = ""
        if open_shell == "alpha":
            insert = "a"
        if open_shell == "beta":
            insert = "b"

        # Format orbital specification
        if mo_type == "occupied":
            if orbital_idx == 0:
                return f"h{insert}"
            return f"h{insert}-{orbital_idx}"
        else:
            if orbital_idx == 0:
                return f"l{insert}"
            return f"l{insert}+{orbital_idx}"

    def _read_output_file(self) -> None:
        """Read the output file from Multiwfn and write the results to the ``results`` dictionary.

        Returns
        -------
        None
        """
        # Check if the output file exists
        _opath = f"{self.__class__.__name__}__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"Multiwfn output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        # Open output file
        with open(_opath, "r") as f:
            multiwfn_output = f.readlines()

        # Find relevant position in the file
        start_idx = None
        for line_idx, line in enumerate(multiwfn_output):
            if all(["Composition of each atom:" in line]):
                start_idx = line_idx + 1

        # Check if start_idx was found
        if start_idx is None:
            self._err = (
                f"output file generated through '{self.__class__.__name__}' does not "
                "contain the requested data; probably the calculation failed. Check the "
                "output file"
            )
            return

        # Save values to results dictionary
        for line in multiwfn_output[start_idx:]:
            if "Atom" in line:
                val = float(line.split(":")[-1].split()[0]) / 100

                # Make negative values zero
                if val < 0:
                    val = 0.0

                atom_idx = int(line.split("Atom")[-1].strip().split("(")[0])
                self.results[atom_idx - 1] = {self.feature_name: val}

            else:
                break


class Multiwfn3DAtomMoContributionOccupiedMulliken(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature "mo_contribution_occupied_mulliken", calculated
    with multiwfn.

    The index of this feature is 264 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_occupied_mulliken`` feature."""
        _read_output = self._run_multiwfn(
            mo_type="occupied",
            orbital_idx=self.homo_minus,
            charge_scheme="mulliken",
            open_shell=False,
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionOccupiedMullikenAlpha(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature "mo_contribution_occupied_mulliken_alpha",
    calculated with multiwfn.

    The index of this feature is 265 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_occupied_mulliken_alpha`` feature."""
        _read_output = self._run_multiwfn(
            mo_type="occupied",
            orbital_idx=self.homo_minus,
            charge_scheme="mulliken",
            open_shell="alpha",
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionOccupiedMullikenBeta(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature "mo_contribution_occupied_mulliken_beta",
    calculated with multiwfn.

    The index of this feature is 266 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_occupied_mulliken_beta`` feature."""
        _read_output = self._run_multiwfn(
            mo_type="occupied",
            orbital_idx=self.homo_minus,
            charge_scheme="mulliken",
            open_shell="beta",
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionOccupiedMullikenRosSchuit(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature "mo_contribution_occupied_mulliken_ros_schuit",
    calculated with multiwfn.

    The index of this feature is 267 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_occupied_mulliken_ros_schuit``
        feature."""
        _read_output = self._run_multiwfn(
            mo_type="occupied",
            orbital_idx=self.homo_minus,
            charge_scheme="mulliken_ros_schuit",
            open_shell=False,
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionOccupiedMullikenRosSchuitAlpha(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature
    "mo_contribution_occupied_mulliken_ros_schuit_alpha", calculated with multiwfn.

    The index of this feature is 268 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_occupied_mulliken_ros_schuit_alpha``
        feature."""
        _read_output = self._run_multiwfn(
            mo_type="occupied",
            orbital_idx=self.homo_minus,
            charge_scheme="mulliken_ros_schuit",
            open_shell="alpha",
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionOccupiedMullikenRosSchuitBeta(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature
    "mo_contribution_occupied_mulliken_ros_schuit_beta", calculated with multiwfn.

    The index of this feature is 269 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_occupied_mulliken_ros_schuit_beta``
        feature."""
        _read_output = self._run_multiwfn(
            mo_type="occupied",
            orbital_idx=self.homo_minus,
            charge_scheme="mulliken_ros_schuit",
            open_shell="beta",
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionOccupiedMullikenStoutPolitzer(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature
    "mo_contribution_occupied_mulliken_stout_politzer", calculated with multiwfn.

    The index of this feature is 270 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_occupied_mulliken_stout_politzer``
        feature."""
        _read_output = self._run_multiwfn(
            mo_type="occupied",
            orbital_idx=self.homo_minus,
            charge_scheme="mulliken_stout_politzer",
            open_shell=False,
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionOccupiedMullikenStoutPolitzerAlpha(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature
    "mo_contribution_occupied_mulliken_stout_politzer_alpha", calculated with multiwfn.

    The index of this feature is 271 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-mo_contribution_occupied_mulliken_stout_politzer_alpha`` feature."""
        _read_output = self._run_multiwfn(
            mo_type="occupied",
            orbital_idx=self.homo_minus,
            charge_scheme="mulliken_stout_politzer",
            open_shell="alpha",
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionOccupiedMullikenStoutPolitzerBeta(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature
    "mo_contribution_occupied_mulliken_stout_politzer_beta", calculated with multiwfn.

    The index of this feature is 272 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-mo_contribution_occupied_mulliken_stout_politzer_beta`` feature."""
        _read_output = self._run_multiwfn(
            mo_type="occupied",
            orbital_idx=self.homo_minus,
            charge_scheme="mulliken_stout_politzer",
            open_shell="beta",
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionUnoccupiedMulliken(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature "mo_contribution_unoccupied_mulliken", calculated
    with multiwfn.

    The index of this feature is 273 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_unoccupied_mulliken`` feature."""
        _read_output = self._run_multiwfn(
            mo_type="unoccupied",
            orbital_idx=self.lumo_plus,
            charge_scheme="mulliken",
            open_shell=False,
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionUnoccupiedMullikenAlpha(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature "mo_contribution_unoccupied_mulliken_alpha",
    calculated with multiwfn.

    The index of this feature is 274 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_unoccupied_mulliken_alpha``
        feature."""
        _read_output = self._run_multiwfn(
            mo_type="unoccupied",
            orbital_idx=self.lumo_plus,
            charge_scheme="mulliken",
            open_shell="alpha",
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionUnoccupiedMullikenBeta(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature "mo_contribution_unoccupied_mulliken_beta",
    calculated with multiwfn.

    The index of this feature is 275 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_unoccupied_mulliken_beta``
        feature."""
        _read_output = self._run_multiwfn(
            mo_type="unoccupied",
            orbital_idx=self.lumo_plus,
            charge_scheme="mulliken",
            open_shell="beta",
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionUnoccupiedMullikenRosSchuit(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature "mo_contribution_unoccupied_mulliken_ros_schuit",
    calculated with multiwfn.

    The index of this feature is 276 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_unoccupied_mulliken_ros_schuit``
        feature."""
        _read_output = self._run_multiwfn(
            mo_type="unoccupied",
            orbital_idx=self.lumo_plus,
            charge_scheme="mulliken_ros_schuit",
            open_shell=False,
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionUnoccupiedMullikenRosSchuitAlpha(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature
    "mo_contribution_unoccupied_mulliken_ros_schuit_alpha", calculated with multiwfn.

    The index of this feature is 277 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-mo_contribution_unoccupied_mulliken_ros_schuit_alpha`` feature."""
        _read_output = self._run_multiwfn(
            mo_type="unoccupied",
            orbital_idx=self.lumo_plus,
            charge_scheme="mulliken_ros_schuit",
            open_shell="alpha",
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionUnoccupiedMullikenRosSchuitBeta(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature
    "mo_contribution_unoccupied_mulliken_ros_schuit_beta", calculated with multiwfn.

    The index of this feature is 278 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_unoccupied_mulliken_ros_schuit_beta``
        feature."""
        _read_output = self._run_multiwfn(
            mo_type="unoccupied",
            orbital_idx=self.lumo_plus,
            charge_scheme="mulliken_ros_schuit",
            open_shell="beta",
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionUnoccupiedMullikenStoutPolitzer(_Multiwfn3DAtomMoContribution):
    """Feature factory for the 3D atom feature
    "mo_contribution_unoccupied_mulliken_stout_politzer", calculated with multiwfn.

    The index of this feature is 279 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``multiwfn3D-atom-mo_contribution_unoccupied_mulliken_stout_politzer``
        feature."""
        _read_output = self._run_multiwfn(
            mo_type="unoccupied",
            orbital_idx=self.lumo_plus,
            charge_scheme="mulliken_stout_politzer",
            open_shell=False,
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionUnoccupiedMullikenStoutPolitzerAlpha(
    _Multiwfn3DAtomMoContribution
):
    """Feature factory for the 3D atom feature
    "mo_contribution_unoccupied_mulliken_stout_politzer_alpha", calculated with multiwfn.

    The index of this feature is 280 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-mo_contribution_unoccupied_mulliken_stout_politzer_alpha`` feature."""
        _read_output = self._run_multiwfn(
            mo_type="unoccupied",
            orbital_idx=self.lumo_plus,
            charge_scheme="mulliken_stout_politzer",
            open_shell="alpha",
        )
        if _read_output is True:
            self._read_output_file()


class Multiwfn3DAtomMoContributionUnoccupiedMullikenStoutPolitzerBeta(
    _Multiwfn3DAtomMoContribution
):
    """Feature factory for the 3D atom feature
    "mo_contribution_unoccupied_mulliken_stout_politzer_beta", calculated with multiwfn.

    The index of this feature is 281 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "multiwfn.orbital" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the
        ``multiwfn3D-atom-mo_contribution_unoccupied_mulliken_stout_politzer_beta`` feature."""
        _read_output = self._run_multiwfn(
            mo_type="unoccupied",
            orbital_idx=self.lumo_plus,
            charge_scheme="mulliken_stout_politzer",
            open_shell="beta",
        )
        if _read_output is True:
            self._read_output_file()
