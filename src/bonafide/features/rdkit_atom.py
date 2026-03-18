"""``RDKit`` features for atoms."""

from rdkit.Chem import EState, rdMolDescriptors, rdPartialCharges

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.helper_functions_chemistry import get_ring_classification


class _Rdkit2DAtomRingInfo(BaseFeaturizer):
    """Parent feature factory for the 2D atom features calculated based on RDKit's
    ``GetRingInfo()``.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def _analyze_rings(self) -> None:
        """Get the RDKit ring information atom features.

        Returns
        -------
        None
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
                        get_ring_classification(mol=self.mol, ring_indices=ring, idx_type="atom")
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
            self.results[self.atom_bond_idx][f"rdkit2D-atom-ring_info_{target_ring_type}"] = res

        all_sizes.sort()
        if all_sizes == []:
            res = "none"
        else:
            res = ",".join([str(s) for s in all_sizes])
        self.results[self.atom_bond_idx]["rdkit2D-atom-ring_info"] = res


class Rdkit2DAtomAtomMapNumber(BaseFeaturizer):
    """Feature factory for the 2D atom feature "atom_map_number", calculated with rdkit.

    The index of this feature is 491 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-atom_map_number`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetAtomMapNum()}


class Rdkit2DAtomAtomicNumber(BaseFeaturizer):
    """Feature factory for the 2D atom feature "atomic_number", calculated with rdkit.

    The index of this feature is 492 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-atomic_number`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetAtomicNum()}


class Rdkit2DAtomChiralTag(BaseFeaturizer):
    """Feature factory for the 2D atom feature "chiral_tag", calculated with rdkit.

    The index of this feature is 493 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-chiral_tag`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: str(atom.GetChiralTag())}


class Rdkit2DAtomDegree(BaseFeaturizer):
    """Feature factory for the 2D atom feature "degree", calculated with rdkit.

    The index of this feature is 494 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-degree`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetDegree()}


class Rdkit2DAtomEstate(BaseFeaturizer):
    """Feature factory for the 2D atom feature "estate", calculated with rdkit.

    The index of this feature is 495 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-estate`` feature."""
        values = list(EState.EStateIndices(self.mol))
        for atom_idx, value in enumerate(values):
            self.results[atom_idx] = {self.feature_name: float(value)}


class Rdkit2DAtomExplicitValence(BaseFeaturizer):
    """Feature factory for the 2D atom feature "explicit_valence", calculated with rdkit.

    The index of this feature is 496 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-explicit_valence`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetExplicitValence()}


class Rdkit2DAtomFormalCharge(BaseFeaturizer):
    """Feature factory for the 2D atom feature "formal_charge", calculated with rdkit.

    The index of this feature is 497 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-formal_charge`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetFormalCharge()}


class Rdkit2DAtomGasteigerCharge(BaseFeaturizer):
    """Feature factory for the 2D atom feature "gasteiger_charge", calculated with rdkit.

    The index of this feature is 498 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-gasteiger_charge`` and ``rdkit2D-atom-gasteiger_h_charge``
        feature.
        """
        rdPartialCharges.ComputeGasteigerCharges(mol=self.mol, throwOnParamFailure=True)
        for atom in self.mol.GetAtoms():
            atom_idx = atom.GetIdx()
            charge = atom.GetProp("_GasteigerCharge")
            h_charge = atom.GetProp("_GasteigerHCharge")
            self.results[atom_idx] = {
                "rdkit2D-atom-gasteiger_charge": float(charge),
                "rdkit2D-atom-gasteiger_h_charge": float(h_charge),
            }


class Rdkit2DAtomGasteigerHCharge(Rdkit2DAtomGasteigerCharge):
    """Feature factory for the 2D atom feature "gasteiger_h_charge", calculated with rdkit.

    The index of this feature is 499 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in Rdkit2DAtomGasteigerCharge


class Rdkit2DAtomHybridization(BaseFeaturizer):
    """Feature factory for the 2D atom feature "hybridization", calculated with rdkit.

    The index of this feature is 500 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-hybridization`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: str(atom.GetHybridization())}


class Rdkit2DAtomImplicitValence(BaseFeaturizer):
    """Feature factory for the 2D atom feature "implicit_valence", calculated with rdkit.

    The index of this feature is 501 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-implicit_valence`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetImplicitValence()}


class Rdkit2DAtomIsAromatic(BaseFeaturizer):
    """Feature factory for the 2D atom feature "is_aromatic", calculated with rdkit.

    The index of this feature is 502 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-is_aromatic`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetIsAromatic()}


class Rdkit2DAtomIsotope(BaseFeaturizer):
    """Feature factory for the 2D atom feature "isotope", calculated with rdkit.

    The index of this feature is 503 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-isotope`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetIsotope()}


class Rdkit2DAtomLabuteAsaContribution(BaseFeaturizer):
    """Feature factory for the 2D atom feature "labute_asa_contribution", calculated with rdkit.

    The index of this feature is 504 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-labute_asa_contribution`` feature."""
        values = list(rdMolDescriptors._CalcLabuteASAContribs(self.mol)[0])
        for atom_idx, value in enumerate(values):
            self.results[atom_idx] = {self.feature_name: value}


class Rdkit2DAtomMass(BaseFeaturizer):
    """Feature factory for the 2D atom feature "mass", calculated with rdkit.

    The index of this feature is 505 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-mass`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetMass()}


class Rdkit2DAtomNExplicitH(BaseFeaturizer):
    """Feature factory for the 2D atom feature "n_explicit_h", calculated with rdkit.

    The index of this feature is 506 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-n_explicit_h`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetNumExplicitHs()}


class Rdkit2DAtomNImplicitH(BaseFeaturizer):
    """Feature factory for the 2D atom feature "n_implicit_h", calculated with rdkit.

    The index of this feature is 507 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-n_implicit_h`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetNumImplicitHs()}


class Rdkit2DAtomNNeighbors(BaseFeaturizer):
    """Feature factory for the 2D atom feature "n_neighbors", calculated with rdkit.

    The index of this feature is 508 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-n_neighbors`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: len(atom.GetNeighbors())}


class Rdkit2DAtomNRadicalElectrons(BaseFeaturizer):
    """Feature factory for the 2D atom feature "n_radical_electrons", calculated with rdkit.

    The index of this feature is 509 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-n_radical_electrons`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetNumRadicalElectrons()}


class Rdkit2DAtomNeighboringAtomsIndices(BaseFeaturizer):
    """Feature factory for the 2D atom feature "neighboring_atoms_indices", calculated with
    rdkit.

    The index of this feature is 510 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-neighboring_atoms_indices`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        neighbor_indices = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        neighbor_indices.sort()
        self.results[self.atom_bond_idx] = {
            self.feature_name: ",".join([str(idx) for idx in neighbor_indices])
        }


class Rdkit2DAtomNeighboringAtomsMapNumbers(BaseFeaturizer):
    """Feature factory for the 2D atom feature "neighboring_atoms_map_numbers", calculated with
    rdkit.

    The index of this feature is 511 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-neighboring_atoms_map_numbers`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        neighbor_map_nums = [neighbor.GetAtomMapNum() for neighbor in atom.GetNeighbors()]
        neighbor_map_nums.sort()
        self.results[self.atom_bond_idx] = {
            self.feature_name: ",".join([str(idx) for idx in neighbor_map_nums])
        }


class Rdkit2DAtomNeighboringBondsIndices(BaseFeaturizer):
    """Feature factory for the 2D atom feature "neighboring_bonds_indices", calculated with
    rdkit.

    The index of this feature is 512 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-neighboring_bonds_indices`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        neighbor_bond_indices = [bond.GetIdx() for bond in atom.GetBonds()]
        neighbor_bond_indices.sort()
        self.results[self.atom_bond_idx] = {
            self.feature_name: ",".join([str(idx) for idx in neighbor_bond_indices])
        }


class Rdkit2DAtomNoImplicit(BaseFeaturizer):
    """Feature factory for the 2D atom feature "no_implicit", calculated with rdkit.

    The index of this feature is 513 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-no_implicit`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetNoImplicit()}


class Rdkit2DAtomRingInfo(_Rdkit2DAtomRingInfo):
    """Feature factory for the 2D atom feature "ring_info", calculated with rdkit.

    The index of this feature is 514 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-ring_info`` feature."""
        self._analyze_rings()


class Rdkit2DAtomRingInfoAromaticCarbocycle(_Rdkit2DAtomRingInfo):
    """Feature factory for the 2D atom feature "ring_info_aromatic_carbocycle", calculated with
    rdkit.

    The index of this feature is 515 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-ring_info_aromatic_carbocycle`` feature."""
        self._analyze_rings()


class Rdkit2DAtomRingInfoAromaticHeterocycle(_Rdkit2DAtomRingInfo):
    """Feature factory for the 2D atom feature "ring_info_aromatic_heterocycle", calculated with
    rdkit.

    The index of this feature is 516 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-ring_info_aromatic_heterocycle`` feature."""
        self._analyze_rings()


class Rdkit2DAtomRingInfoNonaromaticCarbocycle(_Rdkit2DAtomRingInfo):
    """Feature factory for the 2D atom feature "ring_info_nonaromatic_carbocycle", calculated
    with rdkit.

    The index of this feature is 517 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-ring_info_nonaromatic_carbocycle`` feature."""
        self._analyze_rings()


class Rdkit2DAtomRingInfoNonaromaticHeterocycle(_Rdkit2DAtomRingInfo):
    """Feature factory for the 2D atom feature "ring_info_nonaromatic_heterocycle", calculated
    with rdkit.

    The index of this feature is 518 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-ring_info_nonaromatic_heterocycle`` feature."""
        self._analyze_rings()


class Rdkit2DAtomSymbol(BaseFeaturizer):
    """Feature factory for the 2D atom feature "symbol", calculated with rdkit.

    The index of this feature is 529 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-symbol`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetSymbol()}


class Rdkit2DAtomTotalDegree(BaseFeaturizer):
    """Feature factory for the 2D atom feature "total_degree", calculated with rdkit.

    The index of this feature is 530 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-total_degree`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetTotalDegree()}


class Rdkit2DAtomTotalNH(BaseFeaturizer):
    """Feature factory for the 2D atom feature "total_n_h", calculated with rdkit.

    The index of this feature is 531 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-total_n_h`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetTotalNumHs()}


class Rdkit2DAtomTotalValence(BaseFeaturizer):
    """Feature factory for the 2D atom feature "total_valence", calculated with rdkit.

    The index of this feature is 532 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-total_valence`` feature."""
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        self.results[self.atom_bond_idx] = {self.feature_name: atom.GetTotalValence()}


class Rdkit2DAtomTpsaContribution(BaseFeaturizer):
    """Feature factory for the 2D atom feature "tpsa_contribution", calculated with rdkit.

    The index of this feature is 533 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-tpsa_contribution`` feature."""
        values = list(rdMolDescriptors._CalcTPSAContribs(self.mol))
        for atom_idx, value in enumerate(values):
            self.results[atom_idx] = {self.feature_name: value}


class Rdkit2DAtomWildmanCrippenLogpContribution(BaseFeaturizer):
    """Feature factory for the 2D atom feature "wildman_crippen_logp_contribution", calculated
    with rdkit.

    The index of this feature is 534 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-wildman_crippen_logp_contribution`` and
        ``wildman_crippen_mr_contribution`` feature.
        """
        value_pairs = rdMolDescriptors._CalcCrippenContribs(self.mol)
        for atom_idx, value_pair in enumerate(value_pairs):
            self.results[atom_idx] = {
                "rdkit2D-atom-wildman_crippen_logp_contribution": value_pair[0],
                "rdkit2D-atom-wildman_crippen_mr_contribution": value_pair[1],
            }


class Rdkit2DAtomWildmanCrippenMrContribution(Rdkit2DAtomWildmanCrippenLogpContribution):
    """Feature factory for the 2D atom feature "wildman_crippen_mr_contribution", calculated
    with rdkit.

    The index of this feature is 535 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in Rdkit2DAtomWildmanCrippenLogpContribution


class Rdkit3DAtomCoordinates(BaseFeaturizer):
    """Feature factory for the 3D atom feature "coordinates", calculated with rdkit.

    The index of this feature is 556 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.misc" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit3D-atom-coordinates`` feature."""
        pos = self.mol.GetConformer().GetAtomPosition(self.atom_bond_idx)
        atom_coordinates_ = [pos.x, pos.y, pos.z]
        atom_coordinates = ",".join([str(c) for c in atom_coordinates_])
        self.results[self.atom_bond_idx] = {self.feature_name: atom_coordinates}
