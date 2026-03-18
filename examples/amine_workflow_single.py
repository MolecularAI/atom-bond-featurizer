"""Amine featurization workflow for a single compound.

Usage
-----
>>> conda activate bonafide_env
>>> cd examples
>>> python -u amine_workflow_single.py <XYZ input file path> > workflow.log 2>&1
"""

import argparse
import os
from datetime import datetime
from typing import Dict, Tuple, Union

import pandas as pd
from helper_functions import read_csv
from rdkit import Chem

# Input details
MAIN_INPUT_FILE = os.path.join("data", "mol_data_1000.csv")
TARGET_FUNCTIONAL_GROUP = (
    "nitrogen_hydrogen",
    "[N;H1,H2;!+;!-;!$(N-C(=N));!$(N-C(=O));!$(N-C(=S));!$(N-S(=O)=O)]-[H]",
)


def _get_metadata(compound_name: str) -> Tuple[str, int, int]:
    """Get the metadata of a compound from the full dataset CSV file.

    Parameters
    ----------
    compound_name : str
        The name of the compound for which to retrieve the metadata.

    Returns
    -------
    Tuple[str, int, int]
        A tuple containing the SMILES string, formal charge, and multiplicity of the compound.
    """
    df = read_csv(input_file_path=MAIN_INPUT_FILE)
    row = df[df["compound_name"] == compound_name]

    smiles = row["smiles"].values[0]
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    smiles = Chem.MolToSmiles(mol)

    charge = int(row["formal_charge"].values[0])
    multiplicity = int(row["n_unpaired_electrons"].values[0] + 1)

    return smiles, charge, multiplicity


def _analyze_functional_groups(df: pd.DataFrame) -> Union[Dict[str, int], str]:
    """Analyze the functional_group_match feature by checking if the desired function group is
    present in the molecule.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing the atom features of the molecule.

    Returns
    -------
    Union[Dict[str, int], str]
        If the desired functional group is present and there are no avoid groups, a dictionary with
        the indices of the nitrogen and hydrogen atoms is returned. Otherwise, a string indicating
        that the molecule is excluded is returned.
    """
    # Remove atoms that are duplicated due to symmetry
    df = df[df["bonafide2D-atom-is_symmetric_to"] != "_inaccessible"]

    n_targets_found = 0
    avoid_found = False
    nitrogen_atom_idx = None
    hydrogen_atom_idx = None

    # Analysis
    for atom_idx, atom_features in df.iterrows():
        fg_feature = atom_features["bonafide2D-atom-functional_group_match"]
        symbol = atom_features["rdkit2D-atom-symbol"]

        if fg_feature == "no match":
            continue

        groups = list(set(fg_feature.split(",")))
        for group in groups:
            if group == "nitrogen_hydrogen" and symbol == "N":
                n_targets_found += 1
                nitrogen_atom_idx = atom_idx
            elif group == "nitrogen_hydrogen" and symbol == "H":
                hydrogen_atom_idx = atom_idx

            if group == "X_Aliphatic":
                avoid_found = True
                break

    # Filtering
    if n_targets_found == 1 and avoid_found is False:
        return {"nitrogen_atom_idx": nitrogen_atom_idx, "hydrogen_atom_idx": hydrogen_atom_idx}

    return f"Excluded (n_targets_found={n_targets_found}, avoid_found={avoid_found})"


def _get_n_h_bond_idx(df: pd.DataFrame, n_idx: int, h_idx: int) -> int:
    """Get the bond index of the nitrogen-hydrogen bond.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing the bond features of the molecule.
    n_idx : int
        The index of the nitrogen atom.
    h_idx : int
        The index of the hydrogen atom.

    Returns
    -------
    int
        The index of the bond between the nitrogen and hydrogen atoms.
    """
    for bond_idx, row_data in df.iterrows():
        begin_atom_idx = row_data["rdkit2D-bond-begin_atom_index"]
        end_atom_idx = row_data["rdkit2D-bond-end_atom_index"]

        if (begin_atom_idx == n_idx and end_atom_idx == h_idx) or (
            begin_atom_idx == h_idx and end_atom_idx == n_idx
        ):
            return bond_idx


def _run_workflow(
    compound_name: str, input_file_path: str, smiles: str, charge: int, multiplicity: int
) -> None:
    """Run the full amine featurization workflow.

    Parameters
    ----------
    compound_name : str
        The name of the compound.
    input_file_path : str
        The path to the input file for the compound.
    smiles : str
        The SMILES string of the compound.
    charge : int
        The formal charge of the compound.
    multiplicity : int
        The multiplicity of the compound.

    Returns
    -------
    None
    """
    featurizer = AtomBondFeaturizer(f"{compound_name}_00.log")

    featurizer.set_options(("bonafide.symmetry.reduce_to_canonical", True))
    featurizer.set_options(
        (
            "bonafide.functional_group.custom_groups",
            [TARGET_FUNCTIONAL_GROUP],
        )
    )

    featurizer.read_input(input_value=smiles, namespace=compound_name)

    # 27: bonafide2D-atom-functional_group_match
    # 30: bonafide2D-atom-is_symmetric_to
    # 529: rdkit2D-atom-symbol
    featurizer.featurize_atoms(atom_indices="all", feature_indices=[27, 30, 529])
    df = featurizer.return_atom_features()
    df.to_csv(f"FG-analysis_{compound_name}.csv")

    fg_analysis = _analyze_functional_groups(df)
    print(f"Functional group analysis result: {fg_analysis}")
    if isinstance(fg_analysis, str):
        return

    nitrogen_atom_idx = fg_analysis["nitrogen_atom_idx"]
    hydrogen_atom_idx = fg_analysis["hydrogen_atom_idx"]

    # 485: qmdesc2D-atom-fukui_minus
    # 488: qmdesc2D-atom-partial_charge
    # 498: rdkit2D-atom-gasteiger_charge
    # 514: rdkit2D-atom-ring_info
    # 526: rdkit2D-atom-rooted_fingerprint_morgan
    featurizer.featurize_atoms(
        atom_indices=nitrogen_atom_idx, feature_indices=[485, 488, 498, 514, 526]
    )

    # 536: rdkit2D-bond-begin_atom_index
    # 540: rdkit2D-bond-end_atom_index
    featurizer.featurize_bonds(bond_indices="all", feature_indices=[536, 540])
    df = featurizer.return_bond_features()

    n_h_bond_idx = _get_n_h_bond_idx(df, nitrogen_atom_idx, hydrogen_atom_idx)

    # 0: alfabet2D-bond-bond_dissociation_energy
    # 1: alfabet2D-bond-bond_dissociation_free_energy
    # 490: qmdesc2D-bond-bond_order
    featurizer.featurize_bonds(bond_indices=n_h_bond_idx, feature_indices=[0, 1, 490])

    df = featurizer.return_atom_features()
    df.to_csv(f"2D-atom-features_{compound_name}.csv")

    df = featurizer.return_bond_features()
    df.to_csv(f"2D-bond-features_{compound_name}.csv")

    # Calculate 3D features
    featurizer = AtomBondFeaturizer(f"{compound_name}_01.log")
    featurizer.read_input(
        input_value=input_file_path,
        namespace=compound_name,
        input_format="file",
        output_directory=compound_name,
    )

    featurizer.attach_smiles(smiles=smiles, align=False)  # use SMILES atom ordering
    featurizer.set_charge(charge=charge)
    featurizer.set_multiplicity(multiplicity=multiplicity)

    # Run single-point energy calculations
    featurizer.set_options(("psi4.num_threads", 8))
    featurizer.set_options(("psi4.memory", "7 gb"))
    featurizer.calculate_electronic_structure(engine="psi4", redox="all")

    # 98: kallisto3D-atom-partial_charge
    # 177: morfeus3D-atom-fraction_buried_volume
    # 195: morfeus3D-atom-pyramidalization_radhakrishnan
    # 197: morfeus3D-atom-sas_fraction_atom_area
    # 201: multiwfn3D-atom-cdft_condensed_fukui_minus
    # 205: multiwfn3D-atom-cdft_condensed_orbital_weighted_fukui_minus
    # 220: multiwfn3D-atom-cdft_local_nucleophilicity_fmo
    # 221: multiwfn3D-atom-cdft_local_nucleophilicity_redox
    # 237: multiwfn3D-atom-fuzzy_space_atomic_valence
    # 248: multiwfn3D-atom-fuzzy_space_localization_index
    # 264: multiwfn3D-atom-mo_contribution_occupied_mulliken
    # 284: multiwfn3D-atom-partial_charge_cm5
    # 288: multiwfn3D-atom-partial_charge_hirshfeld
    # 296: multiwfn3D-atom-partial_charge_resp_chelpg_two_stage
    # 370: multiwfn3D-atom-surface_average_local_ionization_energy_min
    # 380: multiwfn3D-atom-surface_electrostatic_potential_max
    # 510: rdkit2D-atom-neighboring_atoms_indices
    # 529: rdkit2D-atom-symbol
    # 563: xtb3D-atom-cdft_condensed_fukui_minus
    # 578: xtb3D-atom-cdft_local_nucleophilicity_fmo
    # 579: xtb3D-atom-cdft_local_nucleophilicity_redox
    featurizer.featurize_atoms(
        atom_indices=nitrogen_atom_idx,
        feature_indices=[
            98,
            177,
            195,
            197,
            201,
            205,
            220,
            221,
            237,
            248,
            264,
            284,
            288,
            296,
            370,
            380,
            529,
            563,
            578,
            579,
        ],
    )

    # 431: multiwfn3D-bond-bond_order_wiberg
    # 445: multiwfn3D-bond-intrinsic_bond_strength_index
    # 457: multiwfn3D-bond-topology_bcp_electrostatic_potential
    # 536: rdkit2D-bond-begin_atom_index
    # 540: rdkit2D-bond-end_atom_index
    # 557: rdkit3D-bond-bond_length
    featurizer.featurize_bonds(
        bond_indices=n_h_bond_idx, feature_indices=[431, 445, 457, 536, 540, 557]
    )

    df = featurizer.return_atom_features(atom_indices=nitrogen_atom_idx)
    df.to_csv(f"3D-atom-features_{compound_name}.csv")

    df = featurizer.return_atom_features(atom_indices=nitrogen_atom_idx, reduce=True)
    df.to_csv(f"3D-atom-features_{compound_name}_reduced.csv")

    df = featurizer.return_bond_features(bond_indices=n_h_bond_idx)
    df.to_csv(f"3D-bond-features_{compound_name}.csv")

    df = featurizer.return_bond_features(bond_indices=n_h_bond_idx, reduce=True)
    df.to_csv(f"3D-bond-features_{compound_name}_reduced.csv")


def main() -> None:
    """Main function for featurization.

    It parses the command-line argument, retrieves the metadata for the compound, and runs the
    featurization workflow.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Run amine workflow for a single compound")
    parser.add_argument("file_path", type=str, help="Path to input file for the compound")
    args = parser.parse_args()

    input_file_path = args.file_path
    compound_name = os.path.basename(os.path.dirname(input_file_path))

    smiles, charge, multiplicity = _get_metadata(compound_name)

    print(f"Compound name:    {compound_name}")
    print(f"Input file path:  {input_file_path}")
    print(f"SMILES:           {smiles}")
    print(f"Charge:           {charge}")
    print(f"Multiplicity:     {multiplicity}")
    print()

    print("Starting workflow ...")
    _run_workflow(
        compound_name=compound_name,
        input_file_path=input_file_path,
        smiles=smiles,
        charge=charge,
        multiplicity=multiplicity,
    )
    print("Workflow completed.")


if __name__ == "__main__":
    _start = datetime.now()
    print(f"Start time: {_start}")
    print()

    from bonafide import AtomBondFeaturizer

    main()

    _end = datetime.now()
    print()
    print(f"End time: {_end}")
    print(f"Total time: {_end - _start}")
