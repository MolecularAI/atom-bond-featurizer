"""Features for spiro and bridgehead atoms."""

from typing import Set, Tuple

from bonafide.utils.base_featurizer import BaseFeaturizer


class _Bonafide2DAtomIsSpiroBridgehead(BaseFeaturizer):
    """Parent feature factory for the 2D atom features "is_spiro" and "is_bridgehead"."""

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-is_spiro`` and ``bonafide2D-atom-is_bridgehead``
        feature.
        """
        spiro_atoms, bridgehead_atoms = self._get_spiro_bridgehead()

        for atom in self.mol.GetAtoms():
            atom_idx = atom.GetIdx()

            _is_spiro = False
            _is_bridgehead = False

            if atom_idx in spiro_atoms:
                _is_spiro = True
            if atom_idx in bridgehead_atoms:
                _is_bridgehead = True

            self.results[atom_idx] = {
                "bonafide2D-atom-is_spiro": _is_spiro,
                "bonafide2D-atom-is_bridgehead": _is_bridgehead,
            }

    def _get_spiro_bridgehead(self) -> Tuple[Set[int], Set[int]]:
        """Get the atom indices of spiro and bridgehead atoms in a molecule.

        This is done based on the following heuristics:

        1. If two rings share exactly one atom, then that atom is a spiro atom.
        2. If two rings share more than one atom (intersecting ring), then the shared atoms are
           classified as bridgehead atoms if a given atom of the shared atoms has at least 3
           explicit neighbors that are part of the intersecting rings under consideration.

        Returns
        -------
        Tuple[Set[int], Set[int]]
            A tuple containing two sets: the atom indices of spiro and bridgehead atoms.
        """
        # Get the RDKit ring info analysis, either from the cache or by calculating it
        # Only cache AtomRings() instead of entire GetRingInfo() object to avoid potential memory
        # errors
        _feature_name = "rdkit2d-global-atom_ring_info"
        if _feature_name not in self.global_feature_cache[self.conformer_idx]:
            ring_info = self.mol.GetRingInfo().AtomRings()
            self.global_feature_cache[self.conformer_idx][_feature_name] = ring_info
        else:
            ring_info = self.global_feature_cache[self.conformer_idx][_feature_name]

        ring_atoms = [set(ring) for ring in ring_info]

        # Identify spiro and bridgehead atoms simultaneously
        spiro_atoms = set()
        bridgehead_atoms = set()
        for ring_idx1, ring1 in enumerate(ring_atoms):
            for ring_idx2 in range(ring_idx1 + 1, len(ring_atoms)):
                ring2 = ring_atoms[ring_idx2]
                intersection = ring1.intersection(ring2)

                # No shared atoms between these rings -> continue
                if len(intersection) == 0:
                    continue

                # Exactly one shared atom -> spiro atom
                elif len(intersection) == 1:
                    spiro_atoms.update(intersection)

                # More than one shared atom -> bridgehead atoms
                else:
                    all_ring_indices = ring1.union(ring2)

                    # Check that the shared atoms have at least 3 intersecting ring atom neighbors
                    for atom_idx in intersection:
                        neighbor_indices = [
                            a.GetIdx() for a in self.mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                        ]
                        in_all_ring_indices = [n for n in neighbor_indices if n in all_ring_indices]
                        if len(in_all_ring_indices) >= 3:
                            bridgehead_atoms.add(atom_idx)

        return spiro_atoms, bridgehead_atoms


class Bonafide2DAtomIsSpiro(_Bonafide2DAtomIsSpiroBridgehead):
    """Feature factory for the 2D atom feature "is_spiro", implemented within this package.

    The index of this feature is 29 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature calculation is done in the parent class


class Bonafide2DAtomIsBridgehead(_Bonafide2DAtomIsSpiroBridgehead):
    """Feature factory for the 2D atom feature "is_bridgehead", implemented within this package.

    The index of this feature is 28 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # Feature calculation is done in the parent class
