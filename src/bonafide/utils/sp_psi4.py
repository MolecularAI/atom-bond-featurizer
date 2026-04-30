"""``Psi4`` single-point energy calculation module."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from typing import TYPE_CHECKING, Any, Optional, Tuple

import psi4

from bonafide.utils.base_single_point import BaseSinglePoint
from bonafide.utils.constants import EH_TO_KJ_MOL
from bonafide.utils.helper_functions import clean_up

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class Psi4SP(BaseSinglePoint):
    """Perform a single-point energy calculation with Psi4.

    Parameters
    ----------
    **kwargs : Any
        A dictionary to set class-specific attributes.

    Attributes
    ----------
    basis : str
        The basis set to be used in the calculation.
    charge : int
        The total charge of the molecule.
    CLEAN_SCRATCH_AFTER_CALCULATION : bool
        Whether to clean the scratch directory after the calculation.
    conformer_name : str
        The name of the conformer for which the electronic structure is calculated.
    coordinates : NDArray[np.float64]
        The cartesian coordinates of the conformer.
    elements : NDArray[np.str\_]
        The element symbols of the molecule.
    engine_name : str
        The name of the computational engine used, set to "Psi4".
    maxiter : int
        The maximum number of SCF iterations.
    memory : str
        The amount of memory to be used, e.g., "2 gb".
    method : str
        The quantum chemical method to be used in the calculation.
    multiplicity : int
        The spin multiplicity of the molecule.
    num_threads : int
        The number of threads to be used in the calculation.
    PSI_SCRATCH : str
        The path to the scratch base directory for Psi4 calculations.
    solvent : str
        The solvent to be used in the calculation.
    solvent_model_solver : str
        The solver to be used for the solvent model in the calculation.
    state : str
        The redox state of the molecule, either "n", "n+1", or "n-1".
    """

    PSI_SCRATCH: str
    CLEAN_SCRATCH_AFTER_CALCULATION: bool
    memory: str
    num_threads: int
    basis: str
    maxiter: int
    solvent_model_solver: str
    # Other types are specified in the base class

    def __init__(self, **kwargs: Any) -> None:
        self.engine_name = "Psi4"
        super().__init__(**kwargs)

    def calculate(self, write_el_struc_file: bool) -> Tuple[float, Optional[str]]:
        """Run a single-point energy calculation with Psi4.

        If ``write_el_struc_file`` is ``False``, the molden file path is returned as ``None``.

        Parameters
        ----------
        write_el_struc_file : bool
            Whether to write the calculated electronic structure of the molecule to a file.

        Returns
        -------
        Tuple[float, Optional[str]]
            A tuple containing the electronic energy in kJ/mol and the path to the molden file
            (``None`` if ``write_el_struc_file`` is ``False``).
        """
        # Output file name
        _out_file_name = f"{self.__class__.__name__}__{self.conformer_name}__{self.state}"

        # Get XYZ input string
        structure_input_string = self._get_structure_input_string(
            charge=self.charge,
            multiplicity=self.multiplicity,
            elements=self.elements,
            coordinates=self.coordinates,
        )

        # Set logging level
        _root_logger = logging.getLogger()
        _psi4_loggers = [
            name for name in _root_logger.manager.loggerDict.keys() if name.startswith("psi4")
        ]
        for name in _psi4_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)

        try:
            # Set scratch directory
            scratch_folder_name = f"bonafide-psi4-{self.conformer_name}-{uuid.uuid4().hex[:8]}"
            self._job_scratch = os.path.join(self.PSI_SCRATCH, scratch_folder_name)

            try:
                os.makedirs(name=self._job_scratch, exist_ok=True)
            except Exception as e:
                raise OSError(f"Scratch directory creation at '{self._job_scratch}' failed: {e}")

            if not os.access(path=self._job_scratch, mode=os.W_OK):
                raise OSError(f"Scratch directory at '{self._job_scratch}' is not writable.")

            psi4_io = psi4.core.IOManager.shared_object()
            _previous_scratch = psi4_io.get_default_path()
            psi4_io.set_default_path(self._job_scratch)

            # Set up calculation parameters
            psi4.core.set_output_file(ofname=f"{_out_file_name}.out", append=False)
            psi4.set_memory(inputval=self.memory)
            psi4.core.set_num_threads(self.num_threads)

            # Initialize psi4 molecule
            molecule = psi4.geometry(geom=structure_input_string)

            # Decide if calculation should be restricted or unrestricted
            ref = "uhf" if self.multiplicity > 1 else "rhf"

            # Set options
            psi4.set_options(
                options_dict={
                    "reference": ref,
                    "basis": self.basis,
                    "maxiter": self.maxiter,
                }
            )

            # Handle solvent model
            if self.solvent != "none":
                solvent_string = self._get_solvent_input_string(
                    solvent=self.solvent, solver=self.solvent_model_solver
                )
                psi4.set_options(
                    options_dict={
                        "pcm": "true",
                        "pcm_scf_type": "total",
                        "pcm__input": solvent_string,
                    }
                )

            # Run calculation
            energy, wfn = psi4.energy(self.method, molecule=molecule, return_wfn=True)
            energy = energy * EH_TO_KJ_MOL

            # Get molden file
            if write_el_struc_file is True:
                molden_file_path = os.path.join(
                    os.path.dirname(os.getcwd()), f"{_out_file_name}.molden"
                )
                wfn.write_molden(f"{_out_file_name}.molden")
            else:
                molden_file_path = None

            return energy, molden_file_path

        finally:
            psi4_io = psi4.core.IOManager.shared_object()
            psi4_io.set_default_path(_previous_scratch)
            psi4.core.clean_options()
            self._run_clean_up()

    def _run_clean_up(self) -> None:
        """Remove temporary files from the current working directory and the scratch directory
        after the calculation.

        Returns
        -------
        None
        """
        # Remove job-specific scratch directory if requested
        if self.CLEAN_SCRATCH_AFTER_CALCULATION is True:
            scratch = getattr(self, "_job_scratch", None)
            if scratch:
                shutil.rmtree(path=scratch, ignore_errors=True)

        # Remove .clean file(s) and others from the current working directory
        clean_up(to_be_removed=["*.clean", "*.npz", "PEDRA.OUT__*", "cavity.off__*"])

    @staticmethod
    def _get_structure_input_string(
        charge: int, multiplicity: int, elements: NDArray[np.str_], coordinates: NDArray[np.float64]
    ) -> str:
        """Get the XYZ structure input string for Psi4.

        Parameters
        ----------
        charge : int
            The total charge of the molecule.
        multiplicity : int
            The spin multiplicity of the molecule.
        elements : NDArray[np.str\_]
            The element symbols of the molecule.
        coordinates : NDArray[np.float64]
            The XYZ coordinates of the conformer.

        Returns
        -------
        str
            A string formatted for Psi4 XYZ input.
        """
        xyz = f"{charge} {multiplicity}\n"
        for element, coord in zip(elements, coordinates):
            xyz += f"{element} {coord[0]} {coord[1]} {coord[2]}\n"
        return xyz

    @staticmethod
    def _get_solvent_input_string(solvent: str, solver: str) -> str:
        """Get the input string for the PCM model in Psi4.

        Parameters
        ----------
        solvent : str
            The name of the solvent to be used in the calculation.
        solver : str
            The name of the solver to be used in the calculation.

        Returns
        -------
        str
            A string formatted for the solvent model in Psi4.
        """
        solvent_string = f"""
            Units = Angstrom
            Medium {{
            SolverType = {solver}
            Solvent = {solvent}
            }}

            Cavity {{
            RadiiSet = UFF
            Type = GePol
            Scaling = False
            Area = 0.3
            Mode = Implicit
            }}
        """
        return solvent_string
