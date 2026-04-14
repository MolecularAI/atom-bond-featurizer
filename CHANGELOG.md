# Changelog

## Version 0.1.3

### Changes

-   Update of the information printed in the log file header.

### Fixes

-   The RDKit internal atom property `_ringStereoAtoms` was added to the list of undesired atom
    properties. The absence of this property in the undesired properties list resulted in unexpected
    errors in the `AtomBondFeaturizer.return_atom_features()` method for molecules with chiral ring
    atoms.

## Version 0.1.2

### Changes

-   SD files that contain multiple conformers with different stereochemistry (e.g., E/Z isomerism)
    are now accepted, and a warning message is logged when they are read in. This allows to
    calculate features for stereoisomers together in one conformer ensemble.

### Fixes

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

### Fixes

-   Leave working directory also after failed featurization or single-point energy calculation and
    return to the initial directory.
-   Reset the charge and the spin multiplicity of the molecule vault to the initially cached values
    also after failed single-point energy calculations.

## Version 0.1.0

Initial release.
