"""Calculation of all 2D features for a set of molecules in parallel.

Usage
-----
>>> conda activate bonafide_env
>>> cd examples
>>> python all_2D_features.py
"""

import os
import pickle
from datetime import datetime
from typing import Tuple, Union

import pandas as pd
from helper_functions import parallelizer, read_csv
from rdkit import Chem

# Input details
N_PROCESSORS = 4
INPUT_FILE = os.path.join("data", "mol_data_100.csv")
FEATURE_OUTPUT_FILE = "results_all_2D_features.pkl"


def _run_workflow(
    inp: Tuple[str, str],
) -> Tuple[str, Tuple[Union[pd.DataFrame, str], Union[pd.DataFrame, str]]]:
    """Run the 2D featurization workflow for a single molecule.

    Parameters
    ----------
    inp : Tuple[str, str]
        A tuple containing the compound name and the SMILES string.

    Returns
    -------
    Tuple[str, Tuple[Union[pd.DataFrame, str], Union[pd.DataFrame, str]]]
        A tuple containing the compound name and a tuple of the atom features and bond features.
        The atom features and bond features are returned as pandas DataFrames. If there was an
        error during featurization, the error message is returned as a string instead of a
        DataFrame.
    """
    name = inp[0]
    print(f"Working on '{name}' ...")

    smiles = inp[1]

    from bonafide import AtomBondFeaturizer

    featurizer = AtomBondFeaturizer(f"2D_{name}.log")
    featurizer.read_input(input_value=smiles, namespace=name)

    # Atom featurization
    try:
        featurizer.featurize_atoms(atom_indices="all", feature_indices="all")
    except Exception as e:
        results_atoms = f"Error featurizing atoms: {e}"
    else:
        results_atoms = featurizer.return_atom_features()

    # Bond featurization
    try:
        featurizer.featurize_bonds(bond_indices="all", feature_indices="all")
    except Exception as e:
        results_bonds = f"Error featurizing bonds: {e}"
    else:
        results_bonds = featurizer.return_bond_features()

    return name, (results_atoms, results_bonds)


def main() -> None:
    """Main function for featurization.

    It reads the input CSV file, adds hydrogen atoms to the SMILES strings, and runs the
    featurization workflow in parallel for each molecule. The results are saved to a
    pickle file.

    Returns
    -------
    None
    """
    input_data = read_csv(input_file_path=INPUT_FILE)

    # Add hydrogen atoms to SMILES
    for row_idx, row_data in input_data.iterrows():
        mol = Chem.MolFromSmiles(row_data.smiles)
        mol_with_h = Chem.AddHs(mol)
        input_data.at[row_idx, "smiles"] = Chem.MolToSmiles(mol_with_h)

    # The input CSV file must have a column named "compound_name" and a column named "smiles"
    input_list = [
        (row_data[-1].compound_name, row_data[-1].smiles) for row_data in input_data.iterrows()
    ]

    # Run the featurization workflow in parallel and get the results
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
