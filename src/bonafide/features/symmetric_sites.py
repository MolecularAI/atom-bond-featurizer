"""Identification of symmetry equivalent positions in 2D mol objects."""

import copy
from collections import defaultdict
from typing import Dict, List, Union, cast

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.helper_functions_chemistry import get_symmetric_atom_sites


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
    consider_resonance: bool
    resonance_ALLOW_CHARGE_SEPARATION: bool
    resonance_ALLOW_INCOMPLETE_OCTETS: bool
    resonance_KEKULE_ALL: bool
    resonance_UNCONSTRAINED_ANIONS: bool
    resonance_UNCONSTRAINED_CATIONS: bool

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-is_symmetric_to`` feature."""
        # Get the symmetric atom sites
        sites = get_symmetric_atom_sites(
            mol=self.mol,
            # Configuration settings for Chem.CanonicalRankAtoms
            include_chirality=self.includeChirality,
            include_isotopes=self.includeIsotopes,
            include_atom_maps=self.includeAtomMaps,
            include_chiral_presence=self.includeChiralPresence,
            # Configuration settings for Chem.ResonanceMolSupplier
            consider_resonance=self.consider_resonance,
            resonance_ALLOW_CHARGE_SEPARATION=self.resonance_ALLOW_CHARGE_SEPARATION,
            resonance_ALLOW_INCOMPLETE_OCTETS=self.resonance_ALLOW_INCOMPLETE_OCTETS,
            resonance_KEKULE_ALL=self.resonance_KEKULE_ALL,
            resonance_UNCONSTRAINED_ANIONS=self.resonance_UNCONSTRAINED_ANIONS,
            resonance_UNCONSTRAINED_CATIONS=self.resonance_UNCONSTRAINED_CATIONS,
        )

        # Handle missing indices in initial sites dictionary
        new_sites = cast(Dict[int, Union[List[int], str]], copy.deepcopy(sites))
        for idx_list in sites.values():
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
    consider_resonance: bool
    resonance_ALLOW_CHARGE_SEPARATION: bool
    resonance_ALLOW_INCOMPLETE_OCTETS: bool
    resonance_KEKULE_ALL: bool
    resonance_UNCONSTRAINED_ANIONS: bool
    resonance_UNCONSTRAINED_CATIONS: bool

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-bond-is_symmetric_to`` feature."""
        # Get the symmetric atom sites
        atom_sites = get_symmetric_atom_sites(
            mol=self.mol,
            # Configuration settings for Chem.CanonicalRankAtoms
            include_chirality=self.includeChirality,
            include_isotopes=self.includeIsotopes,
            include_atom_maps=self.includeAtomMaps,
            include_chiral_presence=self.includeChiralPresence,
            # Configuration settings for Chem.ResonanceMolSupplier
            consider_resonance=self.consider_resonance,
            resonance_ALLOW_CHARGE_SEPARATION=self.resonance_ALLOW_CHARGE_SEPARATION,
            resonance_ALLOW_INCOMPLETE_OCTETS=self.resonance_ALLOW_INCOMPLETE_OCTETS,
            resonance_KEKULE_ALL=self.resonance_KEKULE_ALL,
            resonance_UNCONSTRAINED_ANIONS=self.resonance_UNCONSTRAINED_ANIONS,
            resonance_UNCONSTRAINED_CATIONS=self.resonance_UNCONSTRAINED_CATIONS,
        )

        # Get dictionary of symmetry equivalent bond sites defined by the representative atom
        # indices for the symmetry groups of their begin and end atoms
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
        for idx_list in bond_sites.values():
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
