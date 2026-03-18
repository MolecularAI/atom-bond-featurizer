"""Constants."""

from scipy.constants import Avogadro, Boltzmann

# Conversion factor from Hartree to kJ/mol
EH_TO_KJ_MOL = 2625.49964

# Conversion factor from kcal/mol to kJ/mol
KCAL_MOL_TO_KJ_MOL = 4.184

# Conversion factor from kJ/mol to eV
KJ_MOL_TO_EV = 0.01036427

# Universal gas constant in J/K/mol
R = Avogadro * Boltzmann

# Allowed feature types within BONAFIDE
FEATURE_TYPES = ["atom", "bond"]

# Allowed feature and molecule dimensionalities within BONAFIDE
DIMENSIONALITIES = ["2D", "3D"]

# Allowed extensions for electronic structure data files
ELECTRONIC_STRUCTURE_DATA_FILE_EXTENSIONS = ["molden", "fchk"]

# Allowed feature data types within BONAFIDE
DATA_TYPES = ["int", "float", "str", "bool"]

# Allowed input and output types
INPUT_TYPES = ["smiles", "file", "mol_object"]
OUTPUT_TYPES = ["df", "dict", "mol_object"]

# Allowed input file extensions
INPUT_FILE_EXTENSIONS = ["xyz", "sdf"]

# Energy units
ENERGY_UNITS = {
    "kj_mol": [
        "kjmol",
        "kj mol",
        "kj_mol",
        "kj-mol",
        "kj/mol",
        "kjmol-1",
        "kj mol-1",
        "kj_mol-1",
        "kj-mol-1",
        "kj/mol-1",
        "kjmol^-1",
        "kj mol^-1",
        "kj_mol^-1",
        "kj-mol^-1",
        "kj/mol^-1",
        "kjmol_-1",
        "kj mol_-1",
        "kj_mol_-1",
        "kj-mol_-1",
        "kj/mol_-1",
        "kj mol -1",
        "kj_mol -1",
        "kj-mol -1",
        "kj/mol -1",
        "kj_per_mol",
        "kj per mol",
        "kj-per-mol",
        "kj-mol",
        "kilojoulepermol",
        "kilojoule per mol",
        "kilojoule_per_mol",
        "kilojoule-per-mol",
        "kilojoule-mol",
        "kilojoule/mol",
        "kilojoulespermol",
        "kilojoules per mol",
        "kilojoules_per_mol",
        "kilojoules-per-mol",
        "kilojoules-mol",
        "kilojoules/mol",
        "kjpermole",
        "kj per mole",
        "kj_per_mole",
        "kj-per-mole",
        "kj-mole",
        "kilojoulepermole",
        "kilojoule per mole",
        "kilojoule_per_mole",
        "kilojoule-per-mole",
        "kilojoule-mole",
        "kilojoule/mole",
        "kilojoulespermole",
        "kilojoules per mole",
        "kilojoules_per_mole",
        "kilojoules-per-mole",
        "kilojoules-mole",
        "kilojoules/mole",
    ],
    "kcal_mol": [
        "kcalmol",
        "kcal mol",
        "kcal_mol",
        "kcal-mol",
        "kcal/mol",
        "kcalmol-1",
        "kcal mol-1",
        "kcal_mol-1",
        "kcal-mol-1",
        "kcal/mol-1",
        "kcalmol^-1",
        "kcal mol^-1",
        "kcal_mol^-1",
        "kcal-mol^-1",
        "kcal/mol^-1",
        "kcalmol_-1",
        "kcal mol_-1",
        "kcal_mol_-1",
        "kcal-mol_-1",
        "kcal/mol_-1",
        "kcal mol -1",
        "kcal_mol -1",
        "kcal-mol -1",
        "kcal/mol -1",
        "kcal_per_mol",
        "kcal per mol",
        "kcal-per-mol",
        "kcal-mol",
        "kilocalorypermol",
        "kilocalory per mol",
        "kilocalory_per_mol",
        "kilocalory-per-mol",
        "kilocalory-mol",
        "kilocalory/mol",
        "kilocaloriepermol",
        "kilocalorie per mol",
        "kilocalorie_per_mol",
        "kilocalorie-per-mol",
        "kilocalorie-mol",
        "kilocalorie/mol",
        "kilocaloriespermol",
        "kilocalories per mol",
        "kilocalories_per_mol",
        "kilocalories-per-mol",
        "kilocalories-mol",
        "kilocalories/mol",
        "kcalpermole",
        "kcal per mole",
        "kcal_per_mole",
        "kcal-per-mole",
        "kcal-mole",
        "kilocalorypermole",
        "kilocalory per mole",
        "kilocalory_per_mole",
        "kilocalory-per-mole",
        "kilocalory-mole",
        "kilocalory/mole",
        "kilocaloriepermole",
        "kilocalorie per mole",
        "kilocalorie_per_mole",
        "kilocalorie-per-mole",
        "kilocalorie-mole",
        "kilocalorie/mole",
        "kilocaloriespermole",
        "kilocalories per mole",
        "kilocalories_per_mole",
        "kilocalories-per-mole",
        "kilocalories-mole",
        "kilocalories/mole",
    ],
    "eh": ["eh", "e h", "e_h", "e-h", "hartree"],
}

# Methods for determining bonds in mol objects generated from XYZ blocks
DETERMINE_BONDS_METHODS = ["connect_the_dots", "van_der_waals", "hueckel"]

# Electronic structure engines
ELECTRONIC_STRUCTURE_ENGINES = ["psi4", "xtb"]

# Implemented redox states
REDOX_STATES = ["n", "n+1", "n-1", "all"]
REDOX_STATES2 = ["n", "n+1", "n-1"]

# Available electronegativity scales for calculating the oxidation state
ELECTRONEGATIVITY_EN_SCALES = [
    "allen",
    "pauling",
    "ghosh",
    "allred-rochow",
    "cottrell-sutton",
    "gordy",
    "martynov-batsanov",
    "nagle",
    "sanderson",
]

# Available key levels for performing the functional group analysis
FUNCTIONAL_GROUP_KEY_LEVELS = ["l0", "l1", "l2"]

# Available options for calculating SOAP descriptors with DScribe
RBF_METHODS_DSCRIBE_SOAP = ["gto", "polynomial"]
AVERAGE_METHODS_DSCRIBE_SOAP = ["off", "inner", "outer"]

# Available options for calculating LMBTR descriptors with DScribe
GEOMETRY_FUNCTION_METHODS_DSCRIBE_LMBTR = ["distance", "inverse_distance", "angle", "cosine"]
WEIGHTING_FUNCTION_METHODS_DSCRIBE_LMBTR = ["exp", "unity", "inverse_square", "smooth_cutoff"]
NORMALIZATION_METHODS_DSCRIBE_LMBTR = ["none", "l2"]

# Available options for calculating features with kallisto
CNTYPE_METHODS_KALLISTO = ["cov", "exp", "erf"]
VDWTYPE_METHODS_KALLISTO = ["rahm", "truhlar"]

# Available options for calculating features with mendeleev
METHOD_METHODS_MENDELEEV = ["slater", "clementi"]

# Available options for calculating features with MORFEUS
RADII_TYPE_METHODS_MORFEUS_BV_CONE_SOLID_ANGLE = ["alvarez", "bondi", "crc", "truhlar"]
DISTAL_VOLUME_METHODS_MORFEUS_BV = ["sasa"]
RADII_TYPE_METHODS_MORFEUS_DISPERSION = ["alvarez", "bondi", "crc", "rahm", "truhlar"]
METHODS_MORFEUS_LOCAL_FORCE = ["compliance", "local_modes"]
ELECTRONIC_STRUCTURE_DATA_FILE_EXTENSIONS_MORFEUS_LOCAL_FORCE = ["log", "fchk", "hessian"]
RADII_TYPE_METHODS_MORFEUS_SASA = ["bondi", "crc"]
PYRAMIDALIZATION_CALCULATION_METHODS_MORFEUS_PYRAMIDALIZATION = ["distance", "connectivity"]

# Available real space functions as defined in Multiwfn
REAL_SPACE_FUNCTIONS_MULTIWFN = {
    "electron_density": 1,
    "electron_density_gradient_norm": 2,
    "electron_density_laplacian": 3,
    "orbital_wavefunction": 4,
    "spin_density": 5,
    "hamiltonian_kinetic_energy_density": 6,
    "lagrangian_kinetic_energy_density": 7,
    "electrostatic_potential_from_nuclear_charges": 8,
    "electron_localization_function": 9,
    "localized_orbital_locator": 10,
    "local_information_entropy": 11,
    "total_electrostatic_potential": 12,
    "reduced_density_gradient": 13,
    "reduced_density_gradient_promolecular": 14,
    "sign_of_the_second_largest_eigenvalue_electron_density_hessian_matrix": 15,
    "sign_of_the_second_largest_eigenvalue_electron_density_hessian_matrix_promolecular": 16,
    "average_local_ionization_energy": 18,
    "electron_delocalization_range_function": 20,
    "orbital_overlap_distance_function": 21,
    "delta_g_promolecular": 22,
    "delta_g_hirshfeld": 23,
    "interaction_region_indicator": 24,
    "van_der_waals_potential": 25,
}

# Available options for calculating bond analysis features with Multiwfn
IGM_TYPES_MULTIWFN_BOND_ANALYSIS = ["hirshfeld", "promolecular"]
IBIS_GRID_METHODS_MULTIWFN_BOND_ANALYSIS = {"medium": 1, "high": 2, "ultrafine": 3, "perfect": 4}

# Available options for calculating fuzzy space analysis features with Multiwfn
INTEGRATION_GRID_METHODS_MULTIWFN_FUZZY = {"atomic": 1, "molecular": 2}
RADIUS_BECKE_PARTITION_METHODS_MULTIWFN_FUZZY = {
    "csd_tian_lu": -1,
    "csd": 1,
    "pyykko": 2,
    "suresh": 3,
    "hugo": 4,
}
PARTITION_SCHEME_METHODS_MULTIWFN_FUZZY = {"becke": 1, "hirshfeld": 3, "hirshfeld-i": 4}

# Available options for calculating population analysis features with Multiwfn
RADIUS_BECKE_PARTITION_METHODS_MULTIWFN_POPULATION = {
    "csd": 1,
    "csd_tian_lu": 2,
    "pyykko": 3,
    "suresh": 4,
    "hugo": 5,
}
ESP_TYPE_MULTIWFN_POPULATION = {
    "nuclear_electronic": 1,
    "electronic": 2,
    "transition_electronic": 3,
}
ATOMIC_RADII_MULTIWFN_POPULATION = {"automatic": 1, "scaled_uff": 2}
EEM_PARAMETERS_MULTIWFN_POPULATION = {
    "hf_sto-3g_mulliken": 1,
    "b3lyp_6-31g*_chelpg": 2,
    "hf_6-31g*_chelpg": 3,
    "b3lyp_6-311g*_npa": 4,
}

# Available options for calculating CDFT features with Multiwfn
ITERABLE_OPTIONS_MULTIWFN_CDFT = [
    "becke",
    "chelpg",
    "cm5",
    "scaled_cm5",
    "corrected_hirshfeld",
    "hirshfeld",
    "lowdin",
    "merz_kollmann",
    "mulliken",
    "mulliken_bickelhaupt",
    "mulliken_ros_schuit",
    "mulliken_stout_politzer",
    "resp_chelpg_one_stage",
    "resp_chelpg_two_stage",
    "resp_merz_kollmann_one_stage",
    "resp_merz_kollmann_two_stage",
    "vdd",
]

# Available solvents in Psi4
SOLVENTS_PSI4 = [
    "none",
    "acetone",
    "acetonitrile",
    "aniline",
    "benzene",
    "carbon tetrachloride",
    "chlorobenzene",
    "chloroform",
    "cyclohexane",
    "dimethylsulfoxide",
    "ethanol",
    "methanol",
    "methylenechloride",
    "n-heptane",
    "nitromethane",
    "tetrahydrofurane",
    "toluene",
    "water",
]

# Available solvers for the explicit solvent model in Psi4
SOLVENT_MODEL_SOLVERS_PSI4 = ["iefpcm", "cpcm"]

# Strings for checking the xtb version
XTB_VERSION_STRING = "xtb version 6.7.1"

# Available solvent models and solvents in xtb
SOLVENT_MODELS_XTB = ["none", "alpb", "cosmo"]

SOLVENTS_XTB = [
    "acetone",
    "acetonitrile",
    "aniline",
    "benzaldehyde",
    "benzene",
    "ch2cl2",
    "chcl3",
    "cs2",
    "dioxane",
    "dmf",
    "dmso",
    "ether",
    "ethylacetate",
    "furane",
    "hexandecane",
    "hexane",
    "methanol",
    "nitromethane",
    "octanol",
    "phenol",
    "thf",
    "toluene",
    "water",
]

# Available methods in xtb
METHODS_XTB = ["gfn1-xtb", "gfn2-xtb", "gfn0-xtb"]

# Elements of the periodic table
ELEMENT_SYMBOLS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

# Environment variables
PROGRAM_ENVIRONMENT_VARIABLES = {
    "multiwfn": ["OMP_STACKSIZE", "NUM_THREADS"],
    "xtb": [
        "OMP_STACKSIZE",
        "OMP_NUM_THREADS",
        "OMP_MAX_ACTIVE_LEVELS",
        "MKL_NUM_THREADS",
        "XTBHOME",
    ],
}

# Attribute names that cannot be used as keys in the configuration settings
ATTRIBUTE_BLACK_LIST = [
    "_keep_output_files",
    "_periodic_table",
    "_functional_groups_smarts",
    "_state",
    "atom_bond_idx",
    "charge",
    "conformer_idx",
    "mol_vault",
    "conformer_name",
    "coordinates",
    "energy_n",
    "energy_n_plus1",
    "energy_n_minus1",
    "electronic_struc_n",
    "electronic_struc_n_minus1",
    "electronic_struc_n_plus1",
    "electronic_struc_type_n",
    "elements",
    "ensemble_dimensionality",
    "feature_cache",
    "feature_cache_n_minus1",
    "feature_cache_n_plus1",
    "feature_dimensionality",
    "feature_info",
    "feature_name",
    "feature_type",
    "global_feature_cache",
    "mol",
    "multiplicity",
    "extraction_mode",  # used be every feature factory class
    "_err",  # used be the base featurizer class
    "_out",  # used be the base featurizer class
    "results",  # used be the base featurizer class
    "work_dir_name",  # used be the base featurizer class
    "bv_",  # used by the morfeus feature factory
    "sasa_",  # used by the morfeus feature factory
    "dispersion_",  # used by the morfeus feature factory
    "local_force_",  # used by the morfeus feature factory
    "pyra_",  # used by the morfeus feature factory
    "ca_",  # used by the morfeus feature factory
    "sa_",  # used by the morfeus feature factory
]

# Undesired properties of atoms and bonds
UNDESIRED_ATOM_BOND_PROPERTIES = [
    "__computedProps",
    "origNoImplicit",
    "isImplicit",
    "_GasteigerCharge",
    "_GasteigerHCharge",
    "_CIPRank",
    "_NonExplicit3DChirality",
    "_ChiralityPossible",
    "_CIPCode",
    "_ringStereochemCand",
    "_MolFileBondType",
    "_MolFileBondStereo",
    "molParity",
    "_chiralPermutation",
]

UNDESIRED_ATOM_BOND_PROPERTIES2 = ["molAtomMapNumber"]
