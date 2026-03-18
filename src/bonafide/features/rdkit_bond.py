"""RDKit features for bonds."""

import numpy as np
from rdkit.Chem.rdMolTransforms import GetBondLength

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.helper_functions_chemistry import get_ring_classification


class _Rdkit2DBondRingInfo(BaseFeaturizer):
    """Parent feature factory for the 2D bond features calculated based on RDKit's
    ``GetRingInfo()``.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _analyze_rings(self) -> None:
        """Get the RDKit ring information bond features.

        Returns
        -------
        None
        """
        # Get the RDKit ring info analysis, either from the cache or by calculating it
        # Only cache BondRings() instead of entire GetRingInfo() object to avoid potential memory
        # errors
        _feature_name = "rdkit2d-global-bond_ring_info"
        if _feature_name not in self.global_feature_cache[self.conformer_idx]:
            ring_info = self.mol.GetRingInfo().BondRings()
            self.global_feature_cache[self.conformer_idx][_feature_name] = ring_info
        else:
            ring_info = self.global_feature_cache[self.conformer_idx][_feature_name]

        all_sizes = []
        for target_ring_type in [
            "aromatic_carbocycle",
            "aromatic_heterocycle",
            "nonaromatic_carbocycle",
            "nonaromatic_heterocycle",
        ]:
            sizes = []
            for ring in ring_info:
                if self.atom_bond_idx in ring:
                    if (
                        get_ring_classification(mol=self.mol, ring_indices=ring, idx_type="bond")
                        == target_ring_type
                    ):
                        sizes.append(len(ring))

            if sizes == []:
                res = "none"
            else:
                all_sizes.extend(sizes)
                res = ",".join([str(s) for s in sizes])

            if self.atom_bond_idx not in self.results:
                self.results[self.atom_bond_idx] = {}
            self.results[self.atom_bond_idx][f"rdkit2D-bond-ring_info_{target_ring_type}"] = res

        all_sizes.sort()
        if all_sizes == []:
            res = "none"
        else:
            res = ",".join([str(s) for s in all_sizes])
        self.results[self.atom_bond_idx]["rdkit2D-bond-ring_info"] = res


class Rdkit2DBondBeginAtomIndex(BaseFeaturizer):
    """Feature factory for the 2D bond feature "begin_atom_index", calculated with rdkit.

    The index of this feature is 536 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-begin_atom_index`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: bond.GetBeginAtomIdx()}


class Rdkit2DBondBeginAtomMapNumber(BaseFeaturizer):
    """Feature factory for the 2D bond feature "begin_atom_map_number", calculated with rdkit.

    The index of this feature is 537 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-begin_atom_map_number`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: bond.GetBeginAtom().GetAtomMapNum()}


class Rdkit2DBondBondOrder(BaseFeaturizer):
    """Feature factory for the 2D bond feature "bond_order", calculated with rdkit.

    The index of this feature is 538 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-bond_order`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: bond.GetBondTypeAsDouble()}


class Rdkit2DBondBondType(BaseFeaturizer):
    """Feature factory for the 2D bond feature "bond_type", calculated with rdkit.

    The index of this feature is 539 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-bond_type`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: str(bond.GetBondType())}


class Rdkit2DBondEndAtomIndex(BaseFeaturizer):
    """Feature factory for the 2D bond feature "end_atom_index", calculated with rdkit.

    The index of this feature is 540 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-end_atom_index`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: bond.GetEndAtomIdx()}


class Rdkit2DBondEndAtomMapNumber(BaseFeaturizer):
    """Feature factory for the 2D bond feature "end_atom_map_number", calculated with rdkit.

    The index of this feature is 541 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-end_atom_map_number`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: bond.GetEndAtom().GetAtomMapNum()}


class Rdkit2DBondIsAromatic(BaseFeaturizer):
    """Feature factory for the 2D bond feature "is_aromatic", calculated with rdkit.

    The index of this feature is 542 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-is_aromatic`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: bond.GetIsAromatic()}


class Rdkit2DBondIsConjugated(BaseFeaturizer):
    """Feature factory for the 2D bond feature "is_conjugated", calculated with rdkit.

    The index of this feature is 543 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-is_conjugated`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: bond.GetIsConjugated()}


class Rdkit2DBondNNeighbors(BaseFeaturizer):
    """Feature factory for the 2D bond feature "n_neighbors", calculated with rdkit.

    The index of this feature is 544 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-n_neighbors`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        n_neighbors = len(begin_atom.GetNeighbors()) + len(end_atom.GetNeighbors())
        self.results[self.atom_bond_idx] = {
            self.feature_name: n_neighbors - 2
        }  # exclude the bond atoms itself


class Rdkit2DBondNeighboringAtomsIndices(BaseFeaturizer):
    """Feature factory for the 2D bond feature "neighboring_atoms_indices", calculated with
    rdkit.

    The index of this feature is 545 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-neighboring_atoms_indices`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        neighbor_indices = [atom.GetIdx() for atom in begin_atom.GetNeighbors()]
        neighbor_indices.extend([atom.GetIdx() for atom in end_atom.GetNeighbors()])
        neighbor_indices = [
            idx for idx in neighbor_indices if idx not in (begin_atom.GetIdx(), end_atom.GetIdx())
        ]
        neighbor_indices = list(set(neighbor_indices))
        neighbor_indices.sort()

        self.results[self.atom_bond_idx] = {
            self.feature_name: ",".join([str(idx) for idx in neighbor_indices])
        }


class Rdkit2DBondNeighboringAtomsMapNumbers(BaseFeaturizer):
    """Feature factory for the 2D bond feature "neighboring_atoms_map_numbers", calculated with
    rdkit.

    The index of this feature is 546 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-neighboring_atoms_map_numbers`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        _bond_indices = (begin_atom.GetIdx(), end_atom.GetIdx())

        neighbor_dict = {}

        for neighbor in begin_atom.GetNeighbors():
            if neighbor.GetIdx() in _bond_indices:
                continue
            neighbor_dict[neighbor.GetIdx()] = neighbor.GetAtomMapNum()

        for neighbor in end_atom.GetNeighbors():
            if neighbor.GetIdx() in _bond_indices:
                continue
            neighbor_dict[neighbor.GetIdx()] = neighbor.GetAtomMapNum()

        neighbor_map_nums = list(neighbor_dict.values())
        neighbor_map_nums.sort()

        self.results[self.atom_bond_idx] = {
            self.feature_name: ",".join([str(idx) for idx in neighbor_map_nums])
        }


class Rdkit2DBondNeighboringBondsIndices(BaseFeaturizer):
    """Feature factory for the 2D bond feature "neighboring_bonds_indices", calculated with
    rdkit.

    The index of this feature is 547 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-neighboring_bonds_indices`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        neighbor_indices = [bond.GetIdx() for bond in begin_atom.GetBonds()]
        neighbor_indices.extend([bond.GetIdx() for bond in end_atom.GetBonds()])
        neighbor_indices = [idx for idx in neighbor_indices if idx != self.atom_bond_idx]
        neighbor_indices = list(set(neighbor_indices))
        neighbor_indices.sort()

        self.results[self.atom_bond_idx] = {
            self.feature_name: ",".join([str(idx) for idx in neighbor_indices])
        }


class Rdkit2DBondRingInfo(_Rdkit2DBondRingInfo):
    """Feature factory for the 2D bond feature "ring_info", calculated with rdkit.

    The index of this feature is 548 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-ring_info`` feature."""
        self._analyze_rings()


class Rdkit2DBondRingInfoAromaticCarbocycle(_Rdkit2DBondRingInfo):
    """Feature factory for the 2D bond feature "ring_info_aromatic_carbocycle", calculated with
    rdkit.

    The index of this feature is 549 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-ring_info_aromatic_carbocycle`` feature."""
        self._analyze_rings()


class Rdkit2DBondRingInfoAromaticHeterocycle(_Rdkit2DBondRingInfo):
    """Feature factory for the 2D bond feature "ring_info_aromatic_heterocycle", calculated with
    rdkit.

    The index of this feature is 550 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-ring_info_aromatic_heterocycle`` feature."""
        self._analyze_rings()


class Rdkit2DBondRingInfoNonaromaticCarbocycle(_Rdkit2DBondRingInfo):
    """Feature factory for the 2D bond feature "ring_info_nonaromatic_carbocycle", calculated
    with rdkit.

    The index of this feature is 551 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-ring_info_nonaromatic_carbocycle`` feature."""
        self._analyze_rings()


class Rdkit2DBondRingInfoNonaromaticHeterocycle(_Rdkit2DBondRingInfo):
    """Feature factory for the 2D bond feature "ring_info_nonaromatic_heterocycle", calculated
    with rdkit.

    The index of this feature is 552 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-ring_info_nonaromatic_heterocycle`` feature."""
        self._analyze_rings()


class Rdkit2DBondStereo(BaseFeaturizer):
    """Feature factory for the 2D bond feature "stereo", calculated with rdkit.

    The index of this feature is 553 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-stereo`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: str(bond.GetStereo())}


class Rdkit2DBondValenceContributionBeginAtom(BaseFeaturizer):
    """Feature factory for the 2D bond feature "valence_contribution_begin_atom", calculated
    with rdkit.

    The index of this feature is 554 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-valence_contribution_begin_atom`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        begin_atom = bond.GetBeginAtom()
        self.results[self.atom_bond_idx] = {self.feature_name: bond.GetValenceContrib(begin_atom)}


class Rdkit2DBondValenceContributionEndAtom(BaseFeaturizer):
    """Feature factory for the 2D bond feature "valence_contribution_end_atom", calculated with
    rdkit.

    The index of this feature is 555 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-bond-valence_contribution_end_atom`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        end_atom = bond.GetEndAtom()
        self.results[self.atom_bond_idx] = {self.feature_name: bond.GetValenceContrib(end_atom)}


class Rdkit3DBondBondLength(BaseFeaturizer):
    """Feature factory for the 3D bond feature "bond_length", calculated with rdkit.

    The index of this feature is 557 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit3D-bond-bond_length`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        length = GetBondLength(self.mol.GetConformer(0), begin_atom_idx, end_atom_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: round(length, 6)}


class Rdkit3DBondCoordinatesBeginAtom(BaseFeaturizer):
    """Feature factory for the 3D bond feature "coordinates_begin_atom", calculated with rdkit.

    The index of this feature is 558 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit3D-bond-coordinates_begin_atom`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        begin_atom_idx = bond.GetBeginAtomIdx()
        pos = self.mol.GetConformer().GetAtomPosition(begin_atom_idx)
        atom_coordinates_ = [pos.x, pos.y, pos.z]
        atom_coordinates = ",".join([str(round(c, 6)) for c in atom_coordinates_])
        self.results[self.atom_bond_idx] = {self.feature_name: atom_coordinates}


class Rdkit3DBondCoordinatesCenter(BaseFeaturizer):
    """Feature factory for the 3D bond feature "coordinates_center", calculated with rdkit.

    The index of this feature is 559 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit3D-bond-coordinates_center`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()

        begin_pos = self.mol.GetConformer().GetAtomPosition(begin_atom_idx)
        end_pos = self.mol.GetConformer().GetAtomPosition(end_atom_idx)

        begin_coords = np.array([begin_pos.x, begin_pos.y, begin_pos.z])
        end_coords = np.array([end_pos.x, end_pos.y, end_pos.z])

        center_coords = (begin_coords + end_coords) / 2
        center_coordinates = ",".join([str(round(c, 6)) for c in center_coords])
        self.results[self.atom_bond_idx] = {self.feature_name: center_coordinates}


class Rdkit3DBondCoordinatesEndAtom(BaseFeaturizer):
    """Feature factory for the 3D bond feature "coordinates_end_atom", calculated with rdkit.

    The index of this feature is 560 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit3D-bond-coordinates_end_atom`` feature."""
        bond = self.mol.GetBondWithIdx(self.atom_bond_idx)
        end_atom_idx = bond.GetEndAtomIdx()
        pos = self.mol.GetConformer().GetAtomPosition(end_atom_idx)
        atom_coordinates_ = [pos.x, pos.y, pos.z]
        atom_coordinates = ",".join([str(round(c, 6)) for c in atom_coordinates_])
        self.results[self.atom_bond_idx] = {self.feature_name: atom_coordinates}
