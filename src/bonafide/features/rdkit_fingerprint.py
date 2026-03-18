"""RDKit fingerprint features."""

from typing import Any

from rdkit.Chem import rdFingerprintGenerator

from bonafide.utils.base_featurizer import BaseFeaturizer


class Rdkit2DAtomRootedCountFingerprintAtomPair(BaseFeaturizer):
    """Feature factory for the 2D atom feature "rooted_count_fingerprint_atom_pair", calculated
    with rdkit.

    The index of this feature is 519 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.fingerprint" in the _feature_config.toml file.
    """

    countSimulation: bool
    countBounds: Any
    fpSize: int
    includeChirality: bool
    maxDistance: int
    minDistance: int
    use2D: bool

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-rooted_count_fingerprint_atom_pair`` feature."""
        generator = rdFingerprintGenerator.GetAtomPairGenerator(
            minDistance=self.minDistance,
            maxDistance=self.maxDistance,
            includeChirality=self.includeChirality,
            use2D=self.use2D,
            countSimulation=self.countSimulation,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
        )
        numpy_fp = generator.GetCountFingerprintAsNumPy(self.mol, fromAtoms=[self.atom_bond_idx])
        numpy_fp = ",".join([str(x) for x in numpy_fp])
        self.results[self.atom_bond_idx] = {self.feature_name: numpy_fp}


class Rdkit2DAtomRootedCountFingerprintFeatureMorgan(BaseFeaturizer):
    """Feature factory for the 2D atom feature "rooted_count_fingerprint_feature_morgan",
    calculated with rdkit.

    The index of this feature is 520 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.fingerprint" in the _feature_config.toml file.
    """

    countBounds: Any
    countSimulation: bool
    includeChirality: bool
    fpSize: int
    radius: int
    useBondTypes: bool

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-rooted_count_fingerprint_feature_morgan`` feature."""
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            countSimulation=self.countSimulation,
            includeChirality=self.includeChirality,
            useBondTypes=self.useBondTypes,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )
        numpy_fp = generator.GetCountFingerprintAsNumPy(self.mol, fromAtoms=[self.atom_bond_idx])
        numpy_fp = ",".join([str(x) for x in numpy_fp])
        self.results[self.atom_bond_idx] = {self.feature_name: numpy_fp}


class Rdkit2DAtomRootedCountFingerprintMorgan(BaseFeaturizer):
    """Feature factory for the 2D atom feature "rooted_count_fingerprint_morgan", calculated
    with rdkit.

    The index of this feature is 521 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.fingerprint" in the _feature_config.toml file.
    """

    countBounds: Any
    countSimulation: bool
    fpSize: int
    includeChirality: bool
    radius: int
    useBondTypes: bool

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-rooted_count_fingerprint_morgan`` feature."""
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            countSimulation=self.countSimulation,
            includeChirality=self.includeChirality,
            useBondTypes=self.useBondTypes,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
        )
        numpy_fp = generator.GetCountFingerprintAsNumPy(self.mol, fromAtoms=[self.atom_bond_idx])
        numpy_fp = ",".join([str(x) for x in numpy_fp])
        self.results[self.atom_bond_idx] = {self.feature_name: numpy_fp}


class Rdkit2DAtomRootedCountFingerprintRdkit(BaseFeaturizer):
    """Feature factory for the 2D atom feature "rooted_count_fingerprint_rdkit", calculated with
    rdkit.

    The index of this feature is 522 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.fingerprint" in the _feature_config.toml file.
    """

    branchedPaths: bool
    countBounds: Any
    countSimulation: bool
    fpSize: int
    maxPath: int
    minPath: int
    numBitsPerFeature: int
    useBondOrder: bool
    useHs: bool

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-rooted_count_fingerprint_rdkit`` feature."""
        generator = rdFingerprintGenerator.GetRDKitFPGenerator(
            minPath=self.minPath,
            maxPath=self.maxPath,
            useHs=self.useHs,
            branchedPaths=self.branchedPaths,
            useBondOrder=self.useBondOrder,
            countSimulation=self.countSimulation,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
            numBitsPerFeature=self.numBitsPerFeature,
        )
        numpy_fp = generator.GetCountFingerprintAsNumPy(self.mol, fromAtoms=[self.atom_bond_idx])
        numpy_fp = ",".join([str(x) for x in numpy_fp])
        self.results[self.atom_bond_idx] = {self.feature_name: numpy_fp}


class Rdkit2DAtomRootedCountFingerprintTopologicalTorsion(BaseFeaturizer):
    """Feature factory for the 2D atom feature "rooted_count_fingerprint_topological_torsion",
    calculated with rdkit.

    The index of this feature is 523 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.fingerprint" in the _feature_config.toml file.
    """

    countBounds: Any
    countSimulation: bool
    fpSize: int
    includeChirality: bool
    torsionAtomCount: int

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-rooted_count_fingerprint_topological_torsion``
        feature."""
        generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            includeChirality=self.includeChirality,
            torsionAtomCount=self.torsionAtomCount,
            countSimulation=self.countSimulation,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
        )
        numpy_fp = generator.GetCountFingerprintAsNumPy(self.mol, fromAtoms=[self.atom_bond_idx])
        numpy_fp = ",".join([str(x) for x in numpy_fp])
        self.results[self.atom_bond_idx] = {self.feature_name: numpy_fp}


class Rdkit2DAtomRootedFingerprintAtomPair(BaseFeaturizer):
    """Feature factory for the 2D atom feature "rooted_fingerprint_atom_pair", calculated with
    rdkit.

    The index of this feature is 524 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.fingerprint" in the _feature_config.toml file.
    """

    countBounds: Any
    countSimulation: bool
    fpSize: int
    maxDistance: int
    minDistance: int
    includeChirality: bool
    use2D: bool

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-rooted_fingerprint_atom_pair`` feature."""
        generator = rdFingerprintGenerator.GetAtomPairGenerator(
            minDistance=self.minDistance,
            maxDistance=self.maxDistance,
            includeChirality=self.includeChirality,
            use2D=self.use2D,
            countSimulation=self.countSimulation,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
        )
        numpy_fp = generator.GetFingerprintAsNumPy(self.mol, fromAtoms=[self.atom_bond_idx])
        numpy_fp = ",".join([str(x) for x in numpy_fp])
        self.results[self.atom_bond_idx] = {self.feature_name: numpy_fp}


class Rdkit2DAtomRootedFingerprintFeatureMorgan(BaseFeaturizer):
    """Feature factory for the 2D atom feature "rooted_fingerprint_feature_morgan", calculated
    with rdkit.

    The index of this feature is 525 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.fingerprint" in the _feature_config.toml file.
    """

    countBounds: Any
    countSimulation: bool
    includeChirality: bool
    fpSize: int
    radius: int
    useBondTypes: bool

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-rooted_fingerprint_feature_morgan`` feature."""
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            countSimulation=self.countSimulation,
            includeChirality=self.includeChirality,
            useBondTypes=self.useBondTypes,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )
        numpy_fp = generator.GetFingerprintAsNumPy(self.mol, fromAtoms=[self.atom_bond_idx])
        numpy_fp = ",".join([str(x) for x in numpy_fp])
        self.results[self.atom_bond_idx] = {self.feature_name: numpy_fp}


class Rdkit2DAtomRootedFingerprintMorgan(BaseFeaturizer):
    """Feature factory for the 2D atom feature "rooted_fingerprint_morgan", calculated with
    rdkit.

    The index of this feature is 526 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.fingerprint" in the _feature_config.toml file.
    """

    countBounds: Any
    countSimulation: bool
    fpSize: int
    includeChirality: bool
    radius: int
    useBondTypes: bool

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-rooted_fingerprint_morgan`` feature."""
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            countSimulation=self.countSimulation,
            includeChirality=self.includeChirality,
            useBondTypes=self.useBondTypes,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
        )
        numpy_fp = generator.GetFingerprintAsNumPy(self.mol, fromAtoms=[self.atom_bond_idx])
        numpy_fp = ",".join([str(x) for x in numpy_fp])
        self.results[self.atom_bond_idx] = {self.feature_name: numpy_fp}


class Rdkit2DAtomRootedFingerprintRdkit(BaseFeaturizer):
    """Feature factory for the 2D atom feature "rooted_fingerprint_rdkit", calculated with
    rdkit.

    The index of this feature is 527 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.fingerprint" in the _feature_config.toml file.
    """

    branchedPaths: bool
    countBounds: Any
    countSimulation: bool
    fpSize: int
    maxPath: int
    minPath: int
    numBitsPerFeature: int
    useBondOrder: bool
    useHs: bool

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-rooted_fingerprint_rdkit`` feature."""
        generator = rdFingerprintGenerator.GetRDKitFPGenerator(
            minPath=self.minPath,
            maxPath=self.maxPath,
            useHs=self.useHs,
            branchedPaths=self.branchedPaths,
            useBondOrder=self.useBondOrder,
            countSimulation=self.countSimulation,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
            numBitsPerFeature=self.numBitsPerFeature,
        )
        numpy_fp = generator.GetFingerprintAsNumPy(self.mol, fromAtoms=[self.atom_bond_idx])
        numpy_fp = ",".join([str(x) for x in numpy_fp])
        self.results[self.atom_bond_idx] = {self.feature_name: numpy_fp}


class Rdkit2DAtomRootedFingerprintTopologicalTorsion(BaseFeaturizer):
    """Feature factory for the 2D atom feature "rooted_fingerprint_topological_torsion",
    calculated with rdkit.

    The index of this feature is 528 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "rdkit.fingerprint" in the _feature_config.toml file.
    """

    countBounds: Any
    countSimulation: bool
    fpSize: int
    includeChirality: bool
    torsionAtomCount: int

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``rdkit2D-atom-rooted_fingerprint_topological_torsion`` feature."""
        generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            includeChirality=self.includeChirality,
            torsionAtomCount=self.torsionAtomCount,
            countSimulation=self.countSimulation,
            countBounds=self.countBounds,
            fpSize=self.fpSize,
        )
        numpy_fp = generator.GetFingerprintAsNumPy(self.mol, fromAtoms=[self.atom_bond_idx])
        numpy_fp = ",".join([str(x) for x in numpy_fp])
        self.results[self.atom_bond_idx] = {self.feature_name: numpy_fp}
