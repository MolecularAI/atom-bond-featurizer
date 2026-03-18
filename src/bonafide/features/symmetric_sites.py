"""Identification of symmetry equivalent positions in 2D mol objects."""

import copy
from collections import defaultdict
from typing import Dict, List, Union, cast

from rdkit import Chem

from bonafide.utils.base_featurizer import BaseFeaturizer


class Bonafide2DAtomIsSymmetricTo(BaseFeaturizer):
    """Feature factory for the 2D atom feature "is_symmetric_to", implemented within this
    package.

    The index of this feature is 30 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.symmetry" in the _feature_config.toml file.
    """

    includeAtomMaps: bool
    includeChirality: bool
    includeChiralPresence: bool
    includeIsotopes: bool
    reduce_to_canonical: bool

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-is_symmetric_to`` feature."""
        # Rank the atoms based on their canonical ranks (symmetry)
        canonical_rank_list = list(
            Chem.CanonicalRankAtoms(
                mol=self.mol,
                breakTies=False,
                includeChirality=self.includeChirality,
                includeIsotopes=self.includeIsotopes,
                includeAtomMaps=self.includeAtomMaps,
                includeChiralPresence=self.includeChiralPresence,
            )
        )

        # Get dictionary of symmetry equivalent sites
        sites_ = defaultdict(list)
        for atom_idx, rank_idx in enumerate(canonical_rank_list):
            sites_[rank_idx].append(atom_idx)
        sites = {atom_indices[0]: atom_indices for atom_indices in sites_.values()}

        # Handle missing indices in initial sites dictionary
        new_sites = cast(Dict[int, Union[List[int], str]], copy.deepcopy(sites))
        for rank_idx, idx_list in sites.items():
            if len(idx_list) == 1:
                continue

            for atom_idx in idx_list:
                if atom_idx not in sites:
                    if self.reduce_to_canonical is True:
                        new_sites[atom_idx] = "_inaccessible"
                    else:
                        new_sites[atom_idx] = idx_list

        # Write values to the results dictionary
        for atom_idx, idx_list2 in new_sites.items():
            if isinstance(idx_list2, str) and idx_list2 == "_inaccessible":
                self.results[atom_idx] = {self.feature_name: "_inaccessible"}
            else:
                self.results[atom_idx] = {self.feature_name: ",".join([str(i) for i in idx_list2])}


class Bonafide2DBondIsSymmetricTo(BaseFeaturizer):
    """Feature factory for the 2D bond feature "is_symmetric_to", implemented within this
    package.

    The index of this feature is 52 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.symmetry" in the _feature_config.toml file.
    """

    includeAtomMaps: bool
    includeChirality: bool
    includeChiralPresence: bool
    includeIsotopes: bool
    reduce_to_canonical: bool

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-is_symmetric_to`` feature."""
        # Rank the atoms based on their canonical ranks (symmetry)
        canonical_rank_list = list(
            Chem.CanonicalRankAtoms(
                mol=self.mol,
                breakTies=False,
                includeChirality=self.includeChirality,
                includeIsotopes=self.includeIsotopes,
                includeAtomMaps=self.includeAtomMaps,
                includeChiralPresence=self.includeChiralPresence,
            )
        )

        # Get dictionary of symmetry equivalent atom sites
        atom_sites_ = defaultdict(list)
        for atom_idx, rank_idx in enumerate(canonical_rank_list):
            atom_sites_[rank_idx].append(atom_idx)
        atom_sites = {atom_indices[0]: atom_indices for atom_indices in atom_sites_.values()}

        # Get dictionary of symmetry equivalent bond sites defined by the atom rank indices
        # of their begin and end atoms
        bond_sites_ = defaultdict(list)
        for bond in self.mol.GetBonds():
            rank_begin_idx = self._get_rank_idx(rank_dict=atom_sites, idx=bond.GetBeginAtomIdx())
            rank_end_idx = self._get_rank_idx(rank_dict=atom_sites, idx=bond.GetEndAtomIdx())

            bond_id = [rank_begin_idx, rank_end_idx]
            bond_id.sort()
            bond_id_str = "-".join([str(x) for x in bond_id])

            bond_sites_[bond_id_str].append(bond.GetIdx())

        bond_sites = {indices[0]: indices for indices in bond_sites_.values()}

        # Handle missing indices in bond sites dictionary
        new_sites = cast(Dict[int, Union[List[int], str]], copy.deepcopy(bond_sites))
        for rank_idx, idx_list in bond_sites.items():
            if len(idx_list) == 1:
                continue

            for bond_idx in idx_list:
                if bond_idx not in bond_sites:
                    if self.reduce_to_canonical is True:
                        new_sites[bond_idx] = "_inaccessible"
                    else:
                        new_sites[bond_idx] = idx_list

        # Write values to the results dictionary
        for bond_idx, idx_list2 in new_sites.items():
            if isinstance(idx_list2, str) and idx_list2 == "_inaccessible":
                self.results[bond_idx] = {self.feature_name: "_inaccessible"}
            else:
                self.results[bond_idx] = {self.feature_name: ",".join([str(i) for i in idx_list2])}

    @staticmethod
    def _get_rank_idx(rank_dict: Dict[int, List[int]], idx: int) -> int:
        """Get the rank index for a given atom index from the rank dictionary.

        Parameters
        ----------
        rank_dict : Dict[int, List[int]]
            The rank dictionary mapping rank indices to lists of atom indices.
        idx : int
            The atom index for which to find the rank index.

        Returns
        -------
        int
            The rank index corresponding to the given atom index.
        """
        for rank_idx, atom_indices in rank_dict.items():
            if idx in atom_indices:
                return rank_idx

        return -1  # Return -1 if the atom index is not found in the rank dictionary
