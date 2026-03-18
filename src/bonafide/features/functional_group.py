"""Functional group match feature based on predefined SMARTS patterns."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.io_ import read_smarts

if TYPE_CHECKING:
    from rdkit import Chem


class Bonafide2DAtomFunctionalGroupMatch(BaseFeaturizer):
    """Feature factory for the 2D atom feature "functional_group_match", implemented within this
    package.

    The index of this feature is 27 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.functional_group" in the _feature_config.toml file.
    """

    _functional_groups_smarts: Dict[str, List[Tuple[str, Chem.rdchem.Mol]]]
    custom_groups: List[List[str]]
    key_level: str

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-functional_group_match`` feature."""
        # Read in the functional group SMARTS patterns file
        self._read_functional_group_smarts()
        if self._err is not None:
            return

        # Select key level
        smarts_list = self._functional_groups_smarts[self.key_level]

        # Do substructure matching for all SMARTS
        atom_idx_to_key = defaultdict(list)
        for fg_name, smarts_mol in smarts_list:
            matches = self.mol.GetSubstructMatches(smarts_mol)
            for match in matches:
                for atom_idx in match:
                    atom_idx_to_key[atom_idx].append(fg_name)

        # Add "no match" for atoms without any functional group matches
        for atom_idx in range(self.mol.GetNumAtoms()):
            if atom_idx not in atom_idx_to_key:
                atom_idx_to_key[atom_idx].append("no match")

        # Write the features to the results dictionary
        for idx, value in atom_idx_to_key.items():
            res = list(set(value))
            res.sort()
            self.results[idx] = {self.feature_name: ",".join(res)}

    def _read_functional_group_smarts(self) -> None:
        """Read the functional groups SMARTS patterns and store them in a hierarchical dictionary.

        The SMARTS patterns are only read and processed if not already done, i.e. if the
        ``_functional_groups_smarts`` attribute is not empty.

        Returns
        -------
        None
        """
        if self._functional_groups_smarts != {}:
            return

        # Read the functional groups SMARTS from the library file
        _file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "utils",
            "SMARTS_RX.txt",
        )

        # Check if the file exists
        if not os.path.isfile(_file_path):
            self._err = f"Functional groups SMARTS file not found at '{_file_path}'"
            return

        _df = pd.read_csv(
            _file_path,
            sep=" ",
            usecols=[0, 1, 2, 3],
            header=1,
            names=["l0", "l1", "l2", "smarts"],
        )

        # Add custom functional groups
        _df_new = self._add_custom_functional_groups(df=_df)
        if self._err is not None:
            return

        # Validate and format the SMARTS DataFrame
        assert isinstance(_df_new, pd.DataFrame)  # for type checker
        self._validate_and_format_smarts_df(df=_df_new)

    def _add_custom_functional_groups(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Add custom functional groups to the functional groups SMARTS DataFrame.

        Returns
        -------
        Optional[pd.DataFrame]
            The updated functional groups SMARTS DataFrame, or ``None`` if a custom functional
            group name was provided that already exists in the original SMARTS DataFrame.
        """
        for custom_fg in self.custom_groups:
            _name = custom_fg[0]
            _smarts = custom_fg[1]

            # Check if the custom functional group name already exists
            if any(
                [(_name in list(df["l2"])), (_name in list(df["l1"])), (_name in list(df["l0"]))]
            ):
                self._err = f"Custom functional group name '{_name}' already exists"
                return None

            # Append the custom functional group to the DataFrame
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "l0": [_name],
                            "l1": [_name],
                            "l2": [_name],
                            "smarts": [_smarts],
                        }
                    ),
                ],
                ignore_index=True,
            )

        return df

    def _validate_and_format_smarts_df(self, df: pd.DataFrame) -> None:
        """Generate RDKit mol objects from the SMARTS patterns and format the SMARTS DataFrame
        into a hierarchical dictionary.

        Returns
        -------
        None
        """
        # Read and validate SMARTS patterns
        _smarts_mols = []
        for smarts in list(df["smarts"]):
            smarts_mol, error_message = read_smarts(smarts=smarts)

            if error_message is not None:
                self._err = error_message
                return

            _smarts_mols.append(smarts_mol)

        # Write RDKit mol objects to the pandas DataFrame
        df["smarts"] = _smarts_mols

        # Populate _functional_groups_smarts
        for key in [x for x in df.columns if x != "smarts"]:
            self._functional_groups_smarts[key] = [
                (df[key].values[idx], smarts_mol)
                for idx, smarts_mol in enumerate(df["smarts"].values)
            ]
