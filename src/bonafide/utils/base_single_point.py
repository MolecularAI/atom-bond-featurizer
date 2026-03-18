"""Base class for single-point energy calculations with different computational engines."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from bonafide.utils.base_mixin import _BaseMixin
from bonafide.utils.helper_functions import get_function_or_method_name

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from bonafide.utils.molecule_vault import MolVault


class BaseSinglePoint(_BaseMixin):
    """Run single-point energy calculations with different computational engines.

    All conformers in the molecule vault are processed sequentially.

    Attributes
    ----------
    _keep_output_files : bool
        If ``True``, all output files created during the feature calculations are kept. If
        ``False``, they are removed when the calculation is done.
    charge : int
        The total charge of the molecule.
    conformer_name : str
        The name of the conformer.
    coordinates : NDArray[np.float64]
        The cartesian coordinates of the conformer.
    elements : NDArray[np.str\_]
        The element symbols of the molecule.
    engine_name : str
        The name of the computational engine (must be set in the child class).
    mol_vault : MolVault
        The dataclass for storing all relevant data on the molecule.
    multiplicity : int
        The spin multiplicity of the molecule.
    """

    _keep_output_files: bool
    charge: int
    conformer_name: str
    coordinates: NDArray[np.float64]
    elements: NDArray[np.str_]
    engine_name: str
    method: str
    mol_vault: MolVault
    multiplicity: int
    solvent: str
    state: str

    def __init__(self, **kwargs: Any) -> None:
        # Set all attributes required for the single-point energy calculation
        for attr_name, value in kwargs.items():
            setattr(self, attr_name, value)

        # Check if single-point energy class is correctly implemented
        self._check_requirements()

    def _check_requirements(self) -> None:
        """Check if the respective single-point energy class (child class) implements the
        ``calculate()`` method and sets the ``engine_name`` attribute.

        Returns
        -------
        None
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Check if child class has mandatory calculate() method
        method_names = [
            attr
            for attr in dir(self)
            if callable(getattr(self, attr)) is True and not attr.startswith("__")
        ]
        if "calculate" not in method_names:
            _errmsg = (
                "calculate() method must be implemented in engine-specific single-point "
                f"energy class '{self.__class__.__name__}'."
            )
            logging.error(f"'None' | {_loc}()\n{_errmsg}")
            raise NotImplementedError(_errmsg)

        # Check if child class sets engine_name attribute
        if "engine_name" not in vars(self):
            _errmsg = (
                "Attribute 'engine_name' must be set in engine-specific single-point energy class "
                f"'{self.__class__.__name__}'."
            )
            logging.error(f"'None' | {_loc}()\n{_errmsg}")
            raise AttributeError(f"{_loc}(): {_errmsg}")

    def run(
        self, state: str, write_el_struc_file: bool = True
    ) -> Tuple[List[Tuple[Optional[float], str]], List[Optional[str]]]:
        """Run a single-point energy calculation for all conformers of the molecule in the
        molecule vault.

        Parameters
        ----------
        state : str
            The redox state of the molecule to consider, either "n", "n+1", or "n-1".
        write_el_struc_file : bool, optional
            Whether to write the calculated electronic structure of the molecule to an electronic
            structure data file, by default ``True``.

        Returns
        -------
        Tuple[List[Tuple[Optional[float], str]], List[Optional[str]]]
            A tuple containing the data for each conformer:

            * A list of tuples with the electronic energy in kJ/mol (value, unit pair). In case
              the calculation failed, the energy is ``None``.
            * A list of paths to the electronic structure data files. If they were not requested,
              the paths are ``None``.
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Define additional attributes
        # It is ensured that the molecule vault contains the required data for the single-point
        # energy calculation.
        self.elements = self.mol_vault.elements  # type: ignore[assignment]
        self.charge = self.mol_vault.charge  # type: ignore[assignment]
        self.multiplicity = self.mol_vault.multiplicity  # type: ignore[assignment]

        energies = []
        electronic_strucs = []

        # Loop over all conformers in the molecule vault
        for conf_idx, mol in enumerate(self.mol_vault.mol_objects):
            # Setup up working directory and change to it
            self.conformer_name = self.mol_vault.conformer_names[conf_idx]
            _namespace = self.conformer_name[::-1].split("__", 1)[-1][::-1]
            self._setup_work_dir()

            # Initialize results as None
            energy = None
            molden_file_path = None

            # Skip conformers that were labeled as invalid
            if self.mol_vault.is_valid[conf_idx] is False and state == "n":
                logging.warning(
                    f"'{_namespace}' | {_loc}()\nSkipping conformer with index {conf_idx} for "
                    f"single-point energy calculation for state '{state}' with {self.engine_name} "
                    "because it is invalid."
                )

            else:
                logging.info(
                    (
                        f"'{_namespace}' | {_loc}()\nRunning {self.engine_name} single "
                        f"point-energy calculation for conformer with index {conf_idx} for "
                        f"state '{state}'."
                    )
                )

                self.coordinates = mol.GetConformer(0).GetPositions()

                # Try to run the calculation
                try:
                    # self._check_requirements() ensures that the child class implements the
                    # calculate() method; mypy does not recognize this, so we ignore the type
                    # error here
                    energy, molden_file_path = self.calculate(  # type: ignore[attr-defined]
                        write_el_struc_file=write_el_struc_file
                    )
                except Exception as e:
                    _errmsg = (
                        f"Single-point energy calculation with {self.engine_name} failed for "
                        f"conformer with index {conf_idx} for state '{state}': {str(e)}."
                    )
                    if _errmsg.endswith(".."):
                        _errmsg = _errmsg[:-1]
                    logging.error(f"'{_namespace}' | {_loc}()\n{_errmsg}")

            # Collect and store data
            energies.append((energy, "kj_mol"))
            electronic_strucs.append(molden_file_path)

            self._save_output_files()

        return (energies, electronic_strucs)
