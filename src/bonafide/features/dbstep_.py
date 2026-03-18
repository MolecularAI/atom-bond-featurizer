"""Features from ``DBSTEP``."""

import os
import pickle
from typing import List

import dbstep.Dbstep as db

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.io_ import write_xyz_file_from_coordinates_array


class _Dbstep3DAtom(BaseFeaturizer):
    """Parent feature factory for the 3D atom DBSTEP features.

    For details, please refer to the DBSTEP repository (https://github.com/patonlab/DBSTEP,
    last accessed on 09.09.2025).
    """

    addmetals: bool
    exclude: List[int]
    grid: float
    noH: bool
    r: float
    scalevdw: float
    scan: List[float]
    vshell: bool

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _run_dbstep(self) -> None:
        """Run DBSTEP and write the results to the results dictionary.

        Returns
        -------
        None
        """
        # Write input file for DBSTEP
        self._write_input_file()

        # Run DBSTEP
        dbstep_mol = db.dbstep(
            f"{self.conformer_name}.xyz",
            volume=True,
            atom1=self.atom_bond_idx + 1,  # dbstep is 1-indexed
            r=self.r,
            scan=self.scan,
            exclude=self.exclude,
            noH=self.noH,
            addmetals=self.addmetals,
            grid=self.grid,
            vshell=self.vshell,
            scalevdw=self.scalevdw,
            quiet=True,
            commandline=False,
        )
        os.remove(f"{self.conformer_name}.xyz")

        # Format output (bur_vol and bur_shell must be str because they can be multiple values at
        # once, only possible to store them as strings)
        if type(dbstep_mol.bur_vol) == list:
            bur_vol = ",".join([str(round(val / 100, 4)) for val in dbstep_mol.bur_vol])
        else:
            bur_vol = str(round(dbstep_mol.bur_vol / 100, 4))

        if type(dbstep_mol.bur_shell) == list:
            bur_shell = ",".join([str(round(val / 100, 4)) for val in dbstep_mol.bur_shell])
        else:
            bur_shell = str(round(dbstep_mol.bur_shell / 100, 4))

        # Save values to results dictionary
        self.results[self.atom_bond_idx] = {
            "dbstep3D-atom-buried_volume": float(dbstep_mol.occ_vol),
            "dbstep3D-atom-fraction_buried_volume": bur_vol,
            "dbstep3D-atom-fraction_buried_shell_volume": bur_shell,
        }

        # Save DBSTEP object
        with open(f"{self.__class__.__name__}__{self.conformer_name}.pkl", "wb") as f:
            pickle.dump(dbstep_mol, f)

    def _write_input_file(self) -> None:
        """Write an XYZ file as input for DBSTEP.

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


class Dbstep3DAtomBuriedVolume(_Dbstep3DAtom):
    """Feature factory for the 3D atom feature "buried_volume", calculated with dbstep.

    The index of this feature is 90 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "dbstep" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``dbstep3D-atom-buried_volume`` feature."""
        self._run_dbstep()


class Dbstep3DAtomFractionBuriedShellVolume(_Dbstep3DAtom):
    """Feature factory for the 3D atom feature "fraction_buried_shell_volume", calculated with
    dbstep.

    The index of this feature is 91 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "dbstep" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``dbstep3D-atom-fraction_buried_shell_volume`` feature."""
        self._run_dbstep()


class Dbstep3DAtomFractionBuriedVolume(_Dbstep3DAtom):
    """Feature factory for the 3D atom feature "fraction_buried_volume", calculated with dbstep.

    The index of this feature is 92 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "dbstep" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``dbstep3D-atom-fraction_buried_volume`` feature."""
        self._run_dbstep()
