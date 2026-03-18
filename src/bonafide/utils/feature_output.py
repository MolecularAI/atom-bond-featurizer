"""Output formatting after atom and bond featurization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd
from rdkit import Chem

from bonafide.utils.constants import UNDESIRED_ATOM_BOND_PROPERTIES2
from bonafide.utils.helper_functions_output import (
    get_energy_based_reduced_features,
    get_non_energy_based_reduced_features,
)

if TYPE_CHECKING:
    from bonafide.utils.molecule_vault import MolVault


class FeatureOutput:
    """Format the output of the calculated atom or bond features.

    Attributes
    ----------
    _index_name : str
        The name of the index of the pandas DataFrame, either "ATOM_INDEX" or "BOND_INDEX".
    _loc : str
        The name of the current location in the code for logging purposes.
    feature_type : str
        The type of features to return, either "atom" or "bond".
    ignore_invalid : bool
        Whether to ignore invalid conformers during feature reduction.
    indices : List[int]
        The list of atom or bond indices to include.
    mol_vault : MolVault
        The instance of the dataclass for storing all relevant data on the molecule for which
        features were calculated.
    reduce : bool
        Whether to reduce the features to their minimum, maximum, and mean values across all
        conformers. If energies are available, also Boltzmann-averaged values are calculated as
        well as the data for the lowest- and highest-energy conformers.
    """

    def __init__(
        self,
        mol_vault: MolVault,
        indices: List[int],
        feature_type: str,
        reduce: bool,
        ignore_invalid: bool,
        _loc: str,
    ) -> None:
        # Initialize attributes
        self.mol_vault = mol_vault
        self.indices = indices
        self.feature_type = feature_type
        self.reduce = reduce
        self.ignore_invalid = ignore_invalid
        self._loc = _loc

        # Define name of the index of the DataFrame
        if self.feature_type == "atom":
            self._index_name = "ATOM_INDEX"
        if self.feature_type == "bond":
            self._index_name = "BOND_INDEX"

    def get_results(
        self, output_format: str
    ) -> Union[pd.DataFrame, Dict[int, Dict[str, Any]], List[Chem.rdchem.Mol], Chem.rdchem.Mol]:
        """Get the atom and bond features, respectively, in the desired output format.

        Parameters
        ----------
        output_format : str
            The name of the desired output format, can be "df", "dict", or "mol_object".

        Returns
        -------
        Union[pd.DataFrame, Dict[int, Dict[str, Any]], List[Chem.rdchem.Mol], Chem.rdchem.Mol]
            The features in the desired output format.
        """
        # Base case (no feature formatting needed)
        if output_format == "mol_object" and self.reduce is False:
            _mols = self._clear_mols(mols=self.mol_vault.mol_objects)
            _mols = self._fill_missing_features(mols=_mols)
            return _mols

        # Get the features of all conformers in one DataFrame
        combined_df = None
        for conf_idx, mol in enumerate(self.mol_vault.mol_objects):
            # Skip invalid conformers
            if self.mol_vault.is_valid[conf_idx] is False:
                logging.warning(
                    f"'{self.mol_vault.namespace}' | {self._loc}()\nThe conformer with index "
                    f"'{conf_idx}' is invalid. It is ignored during any feature output "
                    "calculation and formatting."
                )
                continue

            # Get feature DataFrame for this conformer and add it to the DataFrame for
            # all conformers
            combined_df = self._get_feature_df(
                mol=mol, conformer_idx=conf_idx, combined_df=combined_df
            )

        # Check if any valid conformers were found
        if combined_df is None:
            _errmsg = "No valid conformers were found in the molecule vault."
            logging.error(f"'{self.mol_vault.namespace}' | {self._loc}()\n{_errmsg}")
            raise ValueError(f"{self._loc}(): {_errmsg}")

        # Clean the DataFrame from undesired properties
        combined_df = combined_df.drop(columns=UNDESIRED_ATOM_BOND_PROPERTIES2, errors="ignore")

        # Get the reduced DataFrame
        assert isinstance(self.mol_vault.size, int)  # for type checker
        if self.reduce is True and self.mol_vault.size > 1:
            processed_df = self._reduce_conformer_data(df=combined_df)
        elif self.reduce is True and self.mol_vault.size == 1:
            logging.warning(
                f"'{self.mol_vault.namespace}' | {self._loc}()\nThe 'reduce' parameter was set to "
                "True but the molecule vault only contains one conformer. Therefore, unreduced "
                "data is returned."
            )
            processed_df = combined_df
        else:
            processed_df = combined_df

        # Postprocess the DataFrame (remove energy related columns)
        processed_df = self._postprocess_df(df=processed_df)

        match output_format:
            # DataFrame output
            case "df":
                return processed_df

            # Dict output
            case "dict":
                # Special case: dict output without reduce and multiple conformers
                if self.reduce is False and self.mol_vault.size > 1:
                    result_dict = {}
                    for idx in processed_df.index.unique():
                        subset = processed_df.loc[[idx]]
                        result_dict[idx] = {
                            col: subset[col].tolist() for col in processed_df.columns
                        }
                    return result_dict

                # Otherwse just return the dict (since this is either a reduced dataframe - one
                # entry per feature - or a single mol object)
                else:
                    return processed_df.to_dict("index")

            # Mol output with reduce (the non-reduce case is handled above)
            case _:
                _mol = self._cast_reduced_props_to_mol(
                    df=processed_df, mol=self.mol_vault.mol_objects[0]
                )
                return _mol

    def _clear_mols(self, mols: List[Chem.rdchem.Mol]) -> List[Chem.rdchem.Mol]:
        """Remove all properties from all atoms or bonds in the given list of molecule objects.

        Parameters
        ----------
        mols : List[Chem.rdchem.Mol]
            The list of RDKit molecule objects to clean.

        Returns
        -------
        List[Chem.rdchem.Mol]
            The list of cleaned RDKit molecule objects.
        """
        _mols = []
        for mol in mols:
            _mol = Chem.Mol(mol)

            # Remove all bond properties
            if self.feature_type == "atom":
                for bond in _mol.GetBonds():
                    for prop in bond.GetPropNames():
                        bond.ClearProp(prop)

            # Remove all atom properties
            if self.feature_type == "bond":
                for atom in _mol.GetAtoms():
                    for prop in atom.GetPropNames():
                        atom.ClearProp(prop)

            _mols.append(_mol)

        return _mols

    def _fill_missing_features(self, mols: List[Chem.rdchem.Mol]) -> List[Chem.rdchem.Mol]:
        """Fill missing features in the given list of molecule objects with ``NaN`` values.

        Parameters
        ----------
        mols : List[Chem.rdchem.Mol]
            The list of RDKit molecule objects to process.

        Returns
        -------
        List[Chem.rdchem.Mol]
            The list of RDKit molecule objects with missing features filled with ``NaN`` values.
        """
        # Get all feature names
        feature_names = set()
        for mol in mols:
            # Atoms
            if self.feature_type == "atom":
                for atom in mol.GetAtoms():
                    feature_names.update(atom.GetPropNames())
            # Bonds
            if self.feature_type == "bond":
                for bond in mol.GetBonds():
                    feature_names.update(bond.GetPropNames())

        # Fill missing features with NaN and
        for mol in mols:
            # Atoms
            if self.feature_type == "atom":
                for atom in mol.GetAtoms():
                    if atom.GetIdx() in self.indices:
                        for feature in feature_names:
                            if atom.HasProp(feature) == 0:
                                atom.SetProp(feature, "NaN")
            # Bonds
            if self.feature_type == "bond":
                for bond in mol.GetBonds():
                    if bond.GetIdx() in self.indices:
                        for feature in feature_names:
                            if bond.HasProp(feature) == 0:
                                bond.SetProp(feature, "NaN")

        # Check if an atom or bond has only NaN features
        for mol in mols:
            # Atoms
            if self.feature_type == "atom":
                for atom in mol.GetAtoms():
                    if atom.GetIdx() in self.indices:
                        if all(
                            [atom.HasProp(f) and atom.GetProp(f) == "NaN" for f in feature_names]
                        ):
                            logging.warning(
                                f"'{self.mol_vault.namespace}' | {self._loc}()\nThe atom with "
                                f"index {atom.GetIdx()} has all features as NaN values. This is "
                                "probably due to the fact that no features were calculated for "
                                "this atom."
                            )
            # Bonds
            if self.feature_type == "bond":
                for bond in mol.GetBonds():
                    if bond.GetIdx() in self.indices:
                        if all(
                            [bond.HasProp(f) and bond.GetProp(f) == "NaN" for f in feature_names]
                        ):
                            logging.warning(
                                f"'{self.mol_vault.namespace}' | {self._loc}()\nThe bond with "
                                f"index {bond.GetIdx()} has all features as NaN values. This is "
                                "probably due to the fact that no features were calculated for "
                                "this bond."
                            )

        return mols

    def _get_feature_df(
        self, mol: Chem.rdchem.Mol, conformer_idx: int, combined_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Get all atom or bond properties as a pandas DataFrame.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
            The RDKit molecule object with calculated features as atom and bond properties.
        conformer_idx : int
            The index of the conformer in the molecule vault.
        combined_df : Optional[pd.DataFrame]
            The DataFrame with the features from all conformers. This is ``None`` if the
            current conformer is the first valid conformer.

        Returns
        -------
        pd.DataFrame
            The pandas DataFrame with the atoms or bonds as rows and the features as columns.
        """
        # Get atom properties
        if self.feature_type == "atom":
            idx_to_prop = {idx: mol.GetAtomWithIdx(idx).GetPropsAsDict() for idx in self.indices}

        # Get bond properties
        if self.feature_type == "bond":
            idx_to_prop = {idx: mol.GetBondWithIdx(idx).GetPropsAsDict() for idx in self.indices}

        # Get DataFrame
        df = pd.DataFrame(idx_to_prop).T
        df.index.name = self._index_name

        # Add additional information to the DataFrame
        assert isinstance(self.mol_vault.size, int)  # for type checker
        if self.mol_vault.size > 1:
            df["conformer_name"] = self.mol_vault.conformer_names[conformer_idx]

            if self.mol_vault.energies_n_read is True:
                df["conformer_energy"] = self.mol_vault.energies_n[conformer_idx][0]

        # Add Boltzmann weights if available
        if all(
            [
                self.mol_vault.boltzmann_weights != (None, [None]),
                self.mol_vault.boltzmann_weights != (),
            ]
        ):
            df["boltzmann_weight"] = self.mol_vault.boltzmann_weights[-1][conformer_idx]  # type: ignore[index, misc]

        # Add to combined DataFrame
        if combined_df is None:
            return df
        else:
            return pd.concat([combined_df, df])

    def _reduce_conformer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce conformer data by calculating various statistics and Boltzmann-weighted averages.

        Parameters
        ----------
        df : pd.DataFrame
            The feature pandas DataFrame containing the data for the individual conformers.

        Returns
        -------
        pd.DataFrame
            The feature pandas DataFrame with the reduced conformer data.
        """
        _exclude_cols = ["conformer_name", "conformer_energy", "boltzmann_weight"]

        # Non-energy-based reduced feature values
        mean_features, min_features, max_features = get_non_energy_based_reduced_features(
            df=df,
            exclude_cols=_exclude_cols,
            feature_type=self.feature_type,
            _namespace=self.mol_vault.namespace,
            _loc=self._loc,
        )
        logging.info(
            f"'{self.mol_vault.namespace}' | {self._loc}()\nCalculation of the mean, min, and max "
            f"feature values across all conformers per {self.feature_type} done."
        )

        # Build the results DataFrame (reduced DataFrame)
        reduced_df = pd.concat(
            [mean_features, min_features, max_features],
            axis=1,
        )
        reduced_df.index.name = self._index_name

        # Energy-based reduced feature values
        if self.mol_vault.energies_n_read is False:
            logging.warning(
                f"'{self.mol_vault.namespace}' | {self._loc}()\nNo energies are available for the "
                "individual conformers. Therefore, the energy-based feature calculations are "
                "skipped and only the min, max, and mean values are returned in the reduced"
                " DataFrame."
            )
            logging.info(
                f"'{self.mol_vault.namespace}' | {self._loc}()\nCalculation of the feature "
                "values of the lowest- and highest-energy conformer and the Boltzmann-weighted "
                f"average across all conformers per {self.feature_type} done."
            )
            return reduced_df

        # Check if any of the conformer energies are NaN, if they are and the user has asked to not
        # ignore these structures, then the energy-based calculations cannot be performed
        elif df["conformer_energy"].isna().any():
            if self.ignore_invalid is True:
                logging.warning(
                    f"'{self.mol_vault.namespace}' | {self._loc}()\nSome conformers are invalid "
                    "and 'ignore_invalid' was set to True. Therefore, Boltzmann-weighted features "
                    "will be calculated only considering the valid conformers."
                )
            else:
                logging.info(
                    f"'{self.mol_vault.namespace}' | {self._loc}()\nSome conformers are invalid "
                    "and 'ignore_invalid' was set to False. Therefore, all energy-based feature "
                    "calculations are skipped and only the min, max, and mean values are returned "
                    "in the reduced DataFrame."
                )
                logging.info(
                    f"'{self.mol_vault.namespace}' | {self._loc}()\nCalculation of the feature "
                    "values of the lowest- and highest-energy conformer and the Boltzmann-weighted "
                    f"average across all conformers per {self.feature_type} done."
                )
                return reduced_df

        # Calculate min energy, max energy, and Boltzmann-weighted average features
        min_e_features, max_e_features, boltzmann_features = get_energy_based_reduced_features(
            df=df,
            exclude_cols=_exclude_cols,
            feature_type=self.feature_type,
            _namespace=self.mol_vault.namespace,
            _loc=self._loc,
        )
        logging.info(
            f"'{self.mol_vault.namespace}' | {self._loc}()\nCalculation of the feature values of "
            "the lowest- and highest-energy conformer and the Boltzmann-weighted average across "
            f"all conformers per {self.feature_type} done."
        )

        # Rebuild the results DataFrame (reduced DataFrame) to have a nicer column order
        reduced_df = pd.concat(
            [
                mean_features,
                min_features,
                max_features,
                min_e_features,
                max_e_features,
                boltzmann_features,
            ],
            axis=1,
        )
        reduced_df.index.name = self._index_name

        return reduced_df

    def _postprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Postprocess the feature DataFrame by removing unneeded columns and check if any atom or
        bond has all features as ``NaN`` values.

        Parameters
        ----------
        df : pd.DataFrame
            The formatted feature pandas DataFrame before postprocessing.

        Returns
        -------
        pd.DataFrame
            The postprocessed feature pandas DataFrame.
        """
        # Check if conformer_energy is in the DataFrame
        if "conformer_energy" in df.columns:
            _energies_all_nan = bool(df["conformer_energy"].isna().all())
        else:
            _energies_all_nan = True

        # Remove columns
        assert isinstance(self.mol_vault.size, int)  # for type checker
        if any([self.mol_vault.size <= 1, self.reduce is True, _energies_all_nan]):
            df = df.drop(["conformer_energy", "boltzmann_weight"], axis=1, errors="ignore")

        # Check for NaN rows
        df_ = df.drop(
            ["conformer_name", "conformer_energy", "boltzmann_weight"], axis=1, errors="ignore"
        )

        nan_indices = df_.index[df_.isna().all(axis=1)].tolist()
        nan_indices = list(set(nan_indices))
        nan_indices.sort()

        for idx in nan_indices:
            logging.warning(
                f"'{self.mol_vault.namespace}' | {self._loc}()\nThe {self.feature_type} with index "
                f"{idx} has all features as NaN values. This is probably due to the fact that no "
                f"features were calculated for this {self.feature_type}."
            )

        return df

    def _cast_reduced_props_to_mol(self, df: pd.DataFrame, mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
        """Cast the features in the reduced DataFrame to atom or bond properties in a molecule
        object.

        The provided RDKit molecule object is copied and cleaned from all properties and
        conformers.

        Parameters
        ----------
        df : pd.DataFrame
            The feature DataFrame containing the reduced data.
        mol : Chem.rdchem.Mol
            The RDKit molecule object to which the features should be added as properties.

        Returns
        -------
        Chem.rdchem.Mol
            The RDKit molecule object with the features added as atom or bond properties.
        """
        _mol = Chem.Mol(mol)

        # Clear properties
        for atom in _mol.GetAtoms():
            for prop in atom.GetPropNames():
                atom.ClearProp(prop)

        for bond in _mol.GetBonds():
            for prop in bond.GetPropNames():
                bond.ClearProp(prop)

        # Remove conformer
        _mol.RemoveAllConformers()

        # Iterate over the feature DataFrame and set properties in the mol object
        for idx, row in df.iterrows():
            for feature_name in df.columns:
                value = row[feature_name]
                dtype = type(value).__name__

                # Atom properties
                if self.feature_type == "atom":
                    obj = _mol.GetAtomWithIdx(idx)

                # Bond properties
                if self.feature_type == "bond":
                    obj = _mol.GetBondWithIdx(idx)

                # Set property
                if "int" in dtype:
                    obj.SetIntProp(feature_name, int(value))
                elif "float" in dtype:
                    obj.SetDoubleProp(feature_name, float(value))
                elif dtype == "str":
                    obj.SetProp(feature_name, value)
                elif dtype == "bool":
                    obj.SetBoolProp(feature_name, value)

        return _mol
