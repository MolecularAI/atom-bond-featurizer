"""``xtb`` single-point energy calculation module."""

import os
from typing import Any, Dict, Optional, Tuple, Union

from bonafide.utils.base_single_point import BaseSinglePoint
from bonafide.utils.constants import EH_TO_KJ_MOL, PROGRAM_ENVIRONMENT_VARIABLES
from bonafide.utils.driver import xtb_driver
from bonafide.utils.helper_functions import clean_up
from bonafide.utils.io_ import write_xyz_file_from_coordinates_array


class XtbSP(BaseSinglePoint):
    """Perform a single-point energy calculation with xtb.

    Parameters
    ----------
    **kwargs : Any
        A dictionary to set class-specific attributes.

    Attributes
    ----------
    acc : float
        The accuracy level for the calculation.
    charge : int
        The total charge of the molecule.
    conformer_name : str
        The name of the conformer for which the electronic structure is calculated.
    coordinates : NDArray[np.float64]
        The cartesian coordinates of the conformer.
    elements : NDArray[np.str\_]
        The element symbols of the molecule.
    engine_name : str
        The name of the computational engine used, set to "xtb".
    etemp : float
        The electronic temperature for the calculation.
    iterations : int
        The maximum number of SCF iterations for the calculation.
    method : str
        The quantum chemical method to be used in the calculation.
    multiplicity : int
        The spin multiplicity of the molecule.
    solvent : str
        The solvent to be used in the calculation.
    solvent_model : str
        The solvent model to be used in the calculation.
    state : str
        The electronic state of the molecule, either "n", "n+1", or "n-1".
    """

    acc: float
    etemp: int
    iterations: int
    solvent_model: str
    # Other types are specified in the base class

    def __init__(self, **kwargs: Any) -> None:
        self.engine_name = "xtb"
        super().__init__(**kwargs)

    def calculate(
        self,
        write_el_struc_file: bool,
        calc_fukui: bool = False,
        calc_ceh: bool = False,
        out_file_name: Optional[str] = None,
    ) -> Tuple[float, Optional[str]]:
        """Run a single-point energy calculation with xtb.

        If ``write_el_struc_file`` is ``False``, the molden file path is returned as ``None``.

        Parameters
        ----------
        write_el_struc_file : bool
            Whether to write the calculated electronic structure of the molecule to a molden file.
        calc_fukui : bool, optional
            Whether to calculate the Fukui indices as implemented in xtb, by default ``False``.
        calc_ceh : bool, optional
            Whether to calculate charge-extended Hueckel charges, by default ``False``.
        out_file_name : Optional[str], optional
            A custom output file name, by default ``None``. If ``None``, it is automatically
            generated.

        Returns
        -------
        Tuple[float, Optional[str]]
            A tuple containing the electronic energy in kJ/mol and the path to the molden file
            (``None`` if ``write_el_struc_file`` is ``False``).
        """

        # Output file name
        if out_file_name is None:
            _out_file_name = f"{self.__class__.__name__}__{self.conformer_name}__{self.state}"
        else:
            _out_file_name = out_file_name

        # Get XYZ input file
        input_file_path = f"{self.conformer_name}.xyz"
        write_xyz_file_from_coordinates_array(
            elements=self.elements, coordinates=self.coordinates, file_path=input_file_path
        )

        # Set up xtb input dictionary
        input_dict: Dict[str, Optional[Union[str, int, float]]] = {
            "input_file_path": input_file_path,
            "output_file_path": f"{_out_file_name}.out",
        }

        # Add method
        if self.method == "gfn2-xtb":
            input_dict["gfn"] = "2"
        elif self.method == "gfn1-xtb":
            input_dict["gfn"] = "1"
        elif self.method == "gfn0-xtb":
            input_dict["gfn"] = "0"

        # Add further options
        input_dict["iterations"] = self.iterations
        input_dict["acc"] = self.acc
        input_dict["etemp"] = self.etemp
        input_dict["chrg"] = self.charge
        input_dict["uhf"] = self.multiplicity - 1

        if self.solvent_model != "none":
            input_dict[self.solvent_model] = self.solvent

        if write_el_struc_file is True:
            input_dict["molden"] = None

        # Request Fukui coefficients
        if calc_fukui is True:
            input_dict["vfukui"] = None

        # Request charge-extended Hueckel charges
        if calc_ceh is True:
            input_dict["ceh"] = None

        # Set up environment variables
        environment_variables = {
            var: getattr(self, var, None) for var in PROGRAM_ENVIRONMENT_VARIABLES["xtb"]
        }

        # Run calculation
        return_code, stderr = xtb_driver(
            input_dict=input_dict, environment_variables=environment_variables
        )

        # Handle error and remove temporary file(s)
        if return_code != 0:
            _exc = f"xtb terminated with error code {return_code}: {stderr}"
            self._run_clean_up()
            raise Exception(_exc)

        # Read output file
        try:
            energy = self._read_xtb_output(f"{_out_file_name}.out")
        except Exception as e:
            _exc = (
                f"Failed to read the electronic energy from the xtb output file: {e}. Check "
                "the output file."
            )
            raise Exception(_exc)

        # Get molden input file
        if write_el_struc_file is True:
            # For the path: anticipate the file in the end being not in the working directory
            # (one step up the directory tree)
            molden_file_path = os.path.join(
                os.path.dirname(os.getcwd()), f"{_out_file_name}.molden"
            )
            os.rename("molden.input", f"{_out_file_name}.molden")
        else:
            molden_file_path = None

        # Remove temporary file(s)
        self._run_clean_up()
        return energy, molden_file_path

    def _read_xtb_output(self, file: str) -> float:
        """Read the electronic energy from the xtb output file.

        Parameters
        ----------
        file : str
            The path to the xtb output file.

        Returns
        -------
        float
            The electronic energy in kJ/mol.
        """
        with open(file, "r") as f:
            xtb_output = f.readlines()
        for line in xtb_output:
            if "TOTAL ENERGY" in line:
                energy = float(line.split()[-3]) * EH_TO_KJ_MOL
                break
        return energy

    @staticmethod
    def _run_clean_up() -> None:
        """Remove temporary files generated during the xtb calculation.

        Returns
        -------
        None
        """
        clean_up(
            to_be_removed=[
                "charges",
                "wbo",
                "xtbrestart",
                "xtbtopo.mol",
                "*.xyz",
                "xtb.cosmo",
                ".sccnotconverged",
            ]
        )
