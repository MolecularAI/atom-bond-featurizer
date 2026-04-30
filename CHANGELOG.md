# Changelog

## Version 0.2.0

### Added

-   A `consider_resonance` configuration option was added to the `bonafide.symmetry` section of the
    `_feature_config.toml` file. The default value is `True`. This influences the
    `bonafide2D-atom-is_symmetric_to` (feature index 30) and `bonafide2D-bond-is_symmetric_to`
    (feature index 52) feature. If set to `True`, atom/bond symmetry will be recognized that is due
    to multiple resonance structures. This means, the two oxygen atoms in a nitro group (NO2,
    `"*[N+](=O)[O-]"`) will be recognized as symmetry-equivalent (same for the two nitrogen-oxygen
    bonds). This then potentially also extends to multiple functional groups; that is, the oxygen
    atoms in 1,4-dinitrobenzene will all be symmetry-equivalent. If set to `False`, the behavior of
    previous versions is reproduced, in which resonance was not considered. Importantly, this does
    not extend to tautomers and open-shell molecules (see https://doi.org/10.1021/acs.jcim.5c00495).
    The consider resonance functionality was implemented based on RDKit's `ResonanceMolSupplier`,
    and the individual configuration options for the enumeration of the resonance forms can also be
    modified if desired (see `_feature_config.toml`). Cases not covered by `ResonanceMolSupplier`
    were implemented by explicit substructure matching. These are:

    | Functional group | SMILES               | SMARTS                         |
    | ---------------- | -------------------- | ------------------------------ |
    | Phosphinate      | `"*P(=O)[O-]"`       | `"[#15](=[#8])-[#8-]"`         |
    | Phosphonate      | `"*P(=O)([O-])[O-]"` | `"[#15](=[#8])(-[#8-])-[#8-]"` |
    | Sulfinate        | `"*S(=O)[O-]"`       | `"[#16](=[#8])-[#8-]"`         |
    | Sulfonate        | `"*S(=O)(=O)[O-]"`   | `"[#16](=[#8])(=[#8])-[#8-]"`  |

-   The handling of Psi4 scratch files was improved. It can now be configured through the
    `psi4.PSI_SCRATCH` (path of the scratch directory, by default `/tmp/`) and
    `psi4.CLEAN_SCRATCH_AFTER_CALCULATION` (whether or not to remove scratch files after the
    calculation completed, by default `True`) configuration options (cf. the `_feature_config.toml`
    file as well as the `print_options()` and `set_options()` methods).

## Version 0.1.3

### Changed

-   Update of the information printed in the log file header.

### Fixed

-   The RDKit internal atom property `_ringStereoAtoms` was added to the list of undesired atom
    properties. The absence of this property in the undesired properties list resulted in unexpected
    errors in the `AtomBondFeaturizer.return_atom_features()` method for molecules with chiral ring
    atoms.

## Version 0.1.2

### Changed

-   SD files that contain multiple conformers with different stereochemistry (e.g., E/Z isomerism)
    are now accepted, and a warning message is logged when they are read in. This allows to
    calculate features for stereoisomers together in one conformer ensemble.

### Fixed

-   Generation of RDKit mol objects from SMILES strings is now performed with `sanitize=True` and
    `removeHs=False`. This ensures proper handling of stereochemistry, which is not achieved with
    first generating the mol object with `sanitize=False` followed by calling
    `Chem.SanitizeMol(mol)`; see
    (https://greglandrum.github.io/rdkit-blog/posts/2025-06-27-sanitization-and-file-parsing.html,
    accessed April 01, 2026).
-   Generation of RDKit mol objects from SD files is now performed with `sanitize=False` and
    `removeHs=False`. The individual conformers contained in the SD file are then sanitized
    individually followed by assigning stereochemistry.

## Version 0.1.1

### Fixed

-   Leave working directory also after failed featurization or single-point energy calculation and
    return to the initial directory.
-   Reset the charge and the spin multiplicity of the molecule vault to the initially cached values
    also after failed single-point energy calculations.

## Version 0.1.0

Initial release.
