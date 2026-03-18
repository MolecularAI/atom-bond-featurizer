"""Calculation of all 3D features for a set of molecules in parallel.

Usage
-----
>>> conda activate bonafide_env
>>> cd examples
>>> python -u all_3D_features.py > all_3D_features.out &

Notes
-----
* Each molecule is expected to have a separate folder in the MOL_DATA_FOLDER directory, named after
  the compound name.
* Each molecule folder should contain the following files:

    - A 3D structure file in XYZ format, named "<compound name>_c00.xyz"
    - An electronic structure file for the actual molecule in FCHK format, named
      "<compound name>_n.fchk"
    - An electronic structure file for the one-electron reduced molecule in FCHK format, named
      "<compound name>_nplus1.fchk"
    - An electronic structure file for the one-electron oxidized molecule in FCHK format, named
      "<compound name>_nminus1.fchk"
    - An energy file in text format, named "<compound name>.energies", containing the energies of
      the three states (n, n+1, n-1) in the following format:
        Energy (n):    -1034.087660871023 Eh
        Energy (n+1):  -1034.112138636586 Eh
        Energy (n-1):  -1033.826744726434 Eh
"""

import os
import pickle
from datetime import datetime
from typing import Tuple, Union

import pandas as pd
from helper_functions import parallelizer, read_csv, read_energies_files
from rdkit import Chem

# Input details
N_PROCESSORS = 4
INPUT_FILE = os.path.join("data", "mol_data_100.csv")
MOL_DATA_FOLDER = ...
FEATURE_OUTPUT_FILE = "results_all_3D_features.pkl"

# Removed features (as they cannot be routinely calculated for atoms of a molecule):
# 170: morfeus3D-atom-cone_angle
# 171: morfeus3D-atom-cone_angle_solid
# 172: morfeus3D-atom-cone_angle_solid_g_parameter
# 173: morfeus3D-atom-cone_tangent_atoms
# 191: morfeus3D-atom-pyramidalization_alpha
# 192: morfeus3D-atom-pyramidalization_alphas
# 193: morfeus3D-atom-pyramidalization_gavrish
# 194: morfeus3D-atom-pyramidalization_neighbor_indices
# 195: morfeus3D-atom-pyramidalization_radhakrishnan
ATOM_FEATURE_INDICES = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    196,
    197,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
    260,
    261,
    262,
    263,
    264,
    265,
    266,
    267,
    268,
    269,
    270,
    271,
    272,
    273,
    274,
    275,
    276,
    277,
    278,
    279,
    280,
    281,
    282,
    283,
    284,
    285,
    286,
    287,
    288,
    289,
    290,
    291,
    292,
    293,
    294,
    295,
    296,
    297,
    298,
    299,
    300,
    301,
    302,
    303,
    304,
    305,
    306,
    307,
    308,
    309,
    310,
    311,
    312,
    313,
    314,
    315,
    316,
    317,
    318,
    319,
    320,
    321,
    322,
    323,
    324,
    325,
    326,
    327,
    328,
    329,
    330,
    331,
    332,
    333,
    334,
    335,
    336,
    337,
    338,
    339,
    340,
    341,
    342,
    343,
    344,
    345,
    346,
    347,
    348,
    349,
    350,
    351,
    352,
    353,
    354,
    355,
    356,
    357,
    358,
    359,
    360,
    361,
    362,
    363,
    364,
    365,
    366,
    367,
    368,
    369,
    370,
    371,
    372,
    373,
    374,
    375,
    376,
    377,
    378,
    379,
    380,
    381,
    382,
    383,
    384,
    385,
    386,
    387,
    388,
    389,
    390,
    391,
    392,
    393,
    394,
    395,
    396,
    397,
    398,
    399,
    400,
    401,
    402,
    403,
    404,
    405,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
    413,
    414,
    415,
    416,
    417,
    418,
    419,
    420,
    421,
    422,
    423,
    424,
    425,
    426,
    484,
    485,
    486,
    487,
    488,
    491,
    492,
    493,
    494,
    495,
    496,
    497,
    498,
    499,
    500,
    501,
    502,
    503,
    504,
    505,
    506,
    507,
    508,
    509,
    510,
    511,
    512,
    513,
    514,
    515,
    516,
    517,
    518,
    519,
    520,
    521,
    522,
    523,
    524,
    525,
    526,
    527,
    528,
    529,
    530,
    531,
    532,
    533,
    534,
    535,
    556,
    561,
    562,
    563,
    564,
    565,
    566,
    567,
    568,
    569,
    570,
    571,
    572,
    573,
    574,
    575,
    576,
    577,
    578,
    579,
    580,
    581,
    582,
    583,
    584,
    585,
    586,
    587,
    588,
    589,
]

# Removed features (as they require data from a frequency calculation):
# 198: morfeus3D-bond-local_force_constant
# 199: morfeus3D-bond-local_frequency
BOND_FEATURES_INDICES = [
    0,
    1,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    427,
    428,
    429,
    430,
    431,
    432,
    433,
    434,
    435,
    436,
    437,
    438,
    439,
    440,
    441,
    442,
    443,
    444,
    445,
    446,
    447,
    448,
    449,
    450,
    451,
    452,
    453,
    454,
    455,
    456,
    457,
    458,
    459,
    460,
    461,
    462,
    463,
    464,
    465,
    466,
    467,
    468,
    469,
    470,
    471,
    472,
    473,
    474,
    475,
    476,
    477,
    478,
    479,
    480,
    481,
    482,
    483,
    489,
    490,
    536,
    537,
    538,
    539,
    540,
    541,
    542,
    543,
    544,
    545,
    546,
    547,
    548,
    549,
    550,
    551,
    552,
    553,
    554,
    555,
    557,
    558,
    559,
    560,
]


def _run_workflow(
    inp: Tuple[str, str, str, int, int, str, str, str, str],
) -> Tuple[str, Union[str, Tuple[Union[pd.DataFrame, str], Union[pd.DataFrame, str]]]]:
    """Run the 2D/3D featurization workflow for a single molecule.

    Parameters
    ----------
    inp : Tuple[str, str, str, int, int, str, str, str, str]
        A tuple with the following content

        * compound name
        * SMILES string
        * Path to the 3D structure file
        * Formal charge
        * Multiplicity
        * Path to the electronic structure file of the actual molecule
        * Path to the electronic structure file of the one-electron reduced molecule
        * Path to the electronic structure file of the one-electron oxidized molecule
        * Path to the energy file containing the energies of the three states

    Returns
    -------
    Tuple[str, Union[str, Tuple[Union[pd.DataFrame, str], Union[pd.DataFrame, str]]]]
        A tuple containing the compound name and a tuple of the atom features and bond features.
        The atom features and bond features are returned as pandas DataFrames. If there was an
        error during featurization, the error message is returned as a string instead of a
        DataFrame.
    """
    name = inp[0]
    print(f"Working on '{name}' ...")

    smiles = inp[1]
    input_file_path = inp[2]
    charge = inp[3]
    multiplicity = inp[4]
    el_file_path_n = inp[5]
    el_file_path_nplus1 = inp[6]
    el_file_path_nminus1 = inp[7]
    energies_file_path = inp[8]

    (energy_n, unit_n), (energy_nplus1, unit_nplus1), (energy_nminus1, unit_nminus1) = (
        read_energies_files(input_file_path=energies_file_path)
    )

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    smiles = Chem.MolToSmiles(mol)

    results_atoms = None
    results_bonds = None

    from bonafide import AtomBondFeaturizer

    featurizer = AtomBondFeaturizer(f"3D_{name}.log")
    featurizer.read_input(input_value=input_file_path, namespace=name, input_format="file")

    # Preprocessing
    try:
        featurizer.attach_smiles(smiles=smiles)

        featurizer.set_charge(charge=charge)
        featurizer.set_multiplicity(multiplicity=multiplicity)

        featurizer.attach_electronic_structure(electronic_structure_data=el_file_path_n, state="n")
        featurizer.attach_electronic_structure(
            electronic_structure_data=el_file_path_nplus1, state="n+1"
        )
        featurizer.attach_electronic_structure(
            electronic_structure_data=el_file_path_nminus1, state="n-1"
        )

        featurizer.attach_energy(energy_data=(energy_n, unit_n), state="n")
        featurizer.attach_energy(energy_data=(energy_nplus1, unit_nplus1), state="n+1")
        featurizer.attach_energy(energy_data=(energy_nminus1, unit_nminus1), state="n-1")

    except Exception as e:
        return name, f"Error preparing molecule: {e}"

    featurizer.set_options(("multiwfn.NUM_THREADS", 2))

    # Atom featurization
    try:
        featurizer.featurize_atoms(atom_indices="all", feature_indices=ATOM_FEATURE_INDICES)
    except Exception as e:
        results_atoms = f"Error featurizing atoms: {e}"
    else:
        results_atoms = featurizer.return_atom_features()

    # Bond featurization
    try:
        featurizer.featurize_bonds(bond_indices="all", feature_indices=BOND_FEATURES_INDICES)
    except Exception as e:
        results_bonds = f"Error featurizing bonds: {e}"
    else:
        results_bonds = featurizer.return_bond_features()

    return name, (results_atoms, results_bonds)


def main() -> None:
    """Main function for featurization.

    It reads the input CSV file, formats the input, and runs the featurization workflow in parallel
    for each molecule. The results are saved to a pickle file.

    Returns
    -------
    None
    """
    input_data = read_csv(input_file_path=INPUT_FILE)

    input_list = [
        (
            # Name of the molecule
            row_data[-1].compound_name,
            # SMILES string
            row_data[-1].smiles,
            # Path to the 3D structure file
            os.path.join(
                MOL_DATA_FOLDER,
                f"{row_data[-1].compound_name}",
                f"{row_data[-1].compound_name}_c00.xyz",
            ),
            # Formal charge
            row_data[-1].formal_charge,
            # Multiplicity
            row_data[-1].n_unpaired_electrons + 1,
            # Paths to electronic structure files
            os.path.join(
                MOL_DATA_FOLDER,
                f"{row_data[-1].compound_name}",
                f"{row_data[-1].compound_name}_n.fchk",
            ),
            os.path.join(
                MOL_DATA_FOLDER,
                f"{row_data[-1].compound_name}",
                f"{row_data[-1].compound_name}_nplus1.fchk",
            ),
            os.path.join(
                MOL_DATA_FOLDER,
                f"{row_data[-1].compound_name}",
                f"{row_data[-1].compound_name}_nminus1.fchk",
            ),
            # Path to the energy file
            os.path.join(
                MOL_DATA_FOLDER,
                f"{row_data[-1].compound_name}",
                f"{row_data[-1].compound_name}.energies",
            ),
        )
        for row_data in input_data.iterrows()
    ]

    all_results = parallelizer(
        worker=_run_workflow, input_list=input_list, n_processors=N_PROCESSORS
    )
    with open(FEATURE_OUTPUT_FILE, "wb") as f:
        pickle.dump(all_results, f)


if __name__ == "__main__":
    _start = datetime.now()
    print(f"Start time: {_start}")
    print()

    main()

    _end = datetime.now()
    print()
    print(f"End time: {_end}")
    print(f"Total time: {_end - _start}")
