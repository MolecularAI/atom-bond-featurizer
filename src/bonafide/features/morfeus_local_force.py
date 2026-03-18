"""Local force constant features from ``MORFEUS``."""

import logging
import pickle
from typing import Optional, Tuple

from morfeus import LocalForce

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.constants import ELECTRONIC_STRUCTURE_DATA_FILE_EXTENSIONS_MORFEUS_LOCAL_FORCE
from bonafide.utils.helper_functions import get_function_or_method_name


class _Morfeus3DBondLocalForce(BaseFeaturizer):
    """Parent feature factory for the 3D bond MORFEUS local force features.

    For details, please refer to the MORFEUS documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    09.09.2025).
    """

    local_force_: LocalForce
    electronic_struc_type_n: str

    imag_cutoff: float
    method: str
    project_imag: bool
    save_hessian: bool

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``morfeus3D-bond-local_force_constant`` and
        ``morfeus3D-bond-local_frequency`` feature.
        """
        # Initial logging because frequency data is required
        _namespace = self.conformer_name[::-1].split("__", 1)[-1][::-1]
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"
        logging.warning(
            f"'{_namespace}' | {_loc}()\nThe calculation of the '{self.feature_name}' feature "
            "requires the presence of data from a frequency calculation in the respective "
            "electronic structure data file (*.log, *.fchk, or *.hessian). Ensure that the "
            "provided file contains this data."
        )

        # Determine program and file type
        program, file_type = self._get_program_filetype()
        if program is None:
            return

        # Initialize local force constant calculation
        assert self.coordinates is not None  # for type checker
        assert isinstance(self.electronic_struc_n, str)  # for type checker
        assert isinstance(file_type, str)  # for type checker
        self.local_force_ = LocalForce(elements=self.elements, coordinates=self.coordinates)
        self.local_force_.load_file(
            file=self.electronic_struc_n, program=program, filetype=file_type
        )

        # Get normal mode analysis if Hessian is provided
        if file_type == "hessian":
            self.local_force_.normal_mode_analysis(save_hessian=self.save_hessian)

        # Add all bonds of the molecule as internal coordinates (if coords are already present,
        # this is not harmful)
        for bond in self.mol.GetBonds():
            self.local_force_.add_internal_coordinate(
                atoms=[bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1]
            )

        # Compute desired data
        if self.method == "local_modes":
            self.local_force_.compute_local(project_imag=self.project_imag, cutoff=self.imag_cutoff)

        if self.method == "compliance":
            self.local_force_.compute_compliance()

        self.local_force_.compute_frequencies()

        # Save the values to the results dictionary
        self._save_values()

        # Save data
        with open(f"Morfeus3DBondLocalForce__{self.conformer_name}.pkl", "wb") as f:
            pickle.dump(self.local_force_, f)

    def _get_program_filetype(self) -> Tuple[Optional[str], Optional[str]]:
        """Determine the program and file type based on the electronic structure file extension.

        Returns
        -------
        Tuple[Optional[str], Optional[str]]
            A tuple containing the program name and file type. If the file type is invalid,
            returns (``None``, ``None``).
        """
        # Check if valid file was provided
        if (
            self.electronic_struc_type_n
            not in ELECTRONIC_STRUCTURE_DATA_FILE_EXTENSIONS_MORFEUS_LOCAL_FORCE
        ):
            self._err = (
                f"Invalid electronic structure data file type: "
                f"'{self.electronic_struc_type_n}'. Supported types are: "
                f"{ELECTRONIC_STRUCTURE_DATA_FILE_EXTENSIONS_MORFEUS_LOCAL_FORCE}"
            )
            return None, None

        # Determine program and file type
        if self.electronic_struc_type_n == "fchk":
            return "gaussian", "fchk"
        elif self.electronic_struc_type_n == "log":
            return "gaussian", "log"
        return "xtb", "hessian"

    def _save_values(self) -> None:
        """Loop over all bonds and get the local force constant and frequency.

        Returns
        -------
        None
        """
        for bond in self.mol.GetBonds():
            _atom1_idx = bond.GetBeginAtomIdx() + 1
            _atom2_idx = bond.GetEndAtomIdx() + 1

            force_constant = self.local_force_.get_local_force_constant(
                atoms=[_atom1_idx, _atom2_idx]
            )
            local_frequency = self.local_force_.get_local_frequency(atoms=[_atom1_idx, _atom2_idx])

            self.results[bond.GetIdx()] = {
                "morfeus3D-bond-local_force_constant": float(force_constant),
                "morfeus3D-bond-local_frequency": float(local_frequency),
            }


class Morfeus3DBondLocalForceConstant(_Morfeus3DBondLocalForce):
    """Feature factory for the 3D bond feature "local_force_constant", calculated with morfeus.

    The index of this feature is 198 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.local_force" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Morfeus3DBondLocalForce


class Morfeus3DBondLocalFrequency(_Morfeus3DBondLocalForce):
    """Feature factory for the 3D bond feature "local_frequency", calculated with morfeus.

    The index of this feature is 199 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "morfeus.local_force" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Morfeus3DBondLocalForce
