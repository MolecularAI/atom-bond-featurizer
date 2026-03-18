"""Features from ``kallisto``."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.driver import kallisto_driver
from bonafide.utils.io_ import write_xyz_file_from_coordinates_array

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class _Kallisto3DAtom(BaseFeaturizer):
    """Parent feature factory for the 3D atom kallisto features.

    For further details, please refer to the Kallisto documentation
    (https://ehjc.gitbook.io/kallisto/, last accessed on 09.09.2025).
    """

    angstrom: bool
    charge: int
    cntype: str
    coordinates: NDArray[np.float64]
    elements: NDArray[np.str_]
    size: List[str]
    vdwtype: str

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def _run_kallisto(self, input_section: List[str]) -> None:
        """Run kallisto.

        Parameters
        ----------
        input_section : List[str]
            The feature-specific arguments passed to kallisto.

        Returns
        -------
        None
        """
        # Run kallisto through the driver function
        stdout, stderr = kallisto_driver(
            input_section=input_section,
            input_file_path=f"{self.conformer_name}.xyz",
            output_file_name=f"{self.__class__.__name__}__{self.conformer_name}",
        )

        # Check for errors and clean up
        if stderr != "":
            self._err = f"stdout: {stdout}, stderr: {stderr}"
        if os.path.isfile(f"{self.conformer_name}.xyz"):
            os.remove(f"{self.conformer_name}.xyz")

    def _write_input_file(self) -> None:
        """Write an XYZ file as input for kallisto.

        Returns
        -------
        None
        """
        assert self.coordinates is not None  # for type checker
        write_xyz_file_from_coordinates_array(
            elements=self.elements,
            coordinates=self.coordinates,
            file_path=self.conformer_name + ".xyz",
        )

    def _read_output_file(self, normalize: bool = False) -> None:
        """Read the output file from kallisto and write the results to the ``results`` dictionary.

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalize the feature values to sum up to 1, by default False.

        Returns
        -------
        None
        """
        # Open output file
        _opath = f"{self.__class__.__name__}__{self.conformer_name}.out"
        if os.path.isfile(_opath) is False:
            self._err = (
                f"kallisto output file '{_opath}' not found; probably the calculation "
                "did not run. Check your input"
            )
            return

        with open(_opath, "r") as f:
            kallisto_output = f.readlines()[2:]  # Skip header lines

        # Read values
        all_values = [float(value.strip()) for value in kallisto_output]
        sum_ = sum(all_values)

        # Normalize if requested
        if normalize:
            all_values = [value / sum_ for value in all_values]

        # Save values to results dictionary
        for idx, value in enumerate(all_values):
            self.results[idx] = {self.feature_name: value}


class Kallisto3DAtomCoordinationNumber(_Kallisto3DAtom):
    """Feature factory for the 3D atom feature "coordination_number", calculated with kallisto.

    The index of this feature is 97 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "kallisto" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``kallisto3D-atom-coordination_number`` feature."""
        self._write_input_file()

        input_section = ["cns"]
        if self.cntype != "cov":
            input_section.extend(["--cntype", self.cntype])
        self._run_kallisto(input_section=input_section)

        self._read_output_file()


class Kallisto3DAtomPartialCharge(_Kallisto3DAtom):
    """Feature factory for the 3D atom feature "partial_charge", calculated with kallisto.

    The index of this feature is 98 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "kallisto" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``kallisto3D-atom-partial_charge`` feature."""
        self._write_input_file()
        self._run_kallisto(input_section=["eeq", "--chrg", str(self.charge)])
        self._read_output_file()


class Kallisto3DAtomPolarizability(_Kallisto3DAtom):
    """Feature factory for the 3D atom feature "polarizability", calculated with kallisto.

    The index of this feature is 99 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "kallisto" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``kallisto3D-atom-polarizability`` feature."""
        self._write_input_file()
        self._run_kallisto(input_section=["alp", "--chrg", str(self.charge)])
        self._read_output_file()


class Kallisto3DAtomProximityShell(_Kallisto3DAtom):
    """Feature factory for the 3D atom feature "proximity_shell", calculated with kallisto.

    The index of this feature is 100 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "kallisto" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``kallisto3D-atom-proximity_shell`` feature."""
        self._write_input_file()
        self._run_kallisto(input_section=["prox", "--size", self.size[0], self.size[1]])
        self._read_output_file()


class Kallisto3DAtomRelativePolarizability(_Kallisto3DAtom):
    """Feature factory for the 3D atom feature "relative_polarizability", calculated with
    kallisto.

    The index of this feature is 101 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "kallisto" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``kallisto3D-atom-relative_polarizability`` feature."""
        self._write_input_file()
        self._run_kallisto(input_section=["alp", "--chrg", str(self.charge)])
        self._read_output_file(normalize=True)


class Kallisto3DAtomVdwRadius(_Kallisto3DAtom):
    """Feature factory for the 3D atom feature "vdw_radius", calculated with kallisto.

    The index of this feature is 102 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "kallisto" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``kallisto3D-atom-vdw_radius`` feature."""
        self._write_input_file()
        input_section = ["vdw", "--chrg", str(self.charge), "--vdwtype", str(self.vdwtype)]
        if self.angstrom is True:
            input_section.append("--angstrom")
        self._run_kallisto(input_section=input_section)
        self._read_output_file()
