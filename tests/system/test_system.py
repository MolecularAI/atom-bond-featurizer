"""End-to-end system tests for feature values.

Notes
-----
This script can accept two optional command line arguments:

* ``--input_file``: Name of the input file to be used for testing. If not provided, all available
  input files will be used. In order to include an input file, it must be registered in
  ``system_test_data_files.json`` (ground truth data file and already passed data file). The
  input files are located in /tests/system/data/.
* ``--recalc_all``: If provided, all features will be recalculated and compared to the expected
  values. If not provided, only features that have not yet passed will be explicitly tested.

Important: a new input strings must be registered in ``input_string_metadata.json``.

Examples
--------
* Run all system tests with all registered input files (default):
>>> pytest -s test_system.py

* Select a specific input file:
>>> pytest -s test_system.py --input_file system_test_data2.json

* Force the recalculation of all feature values (for all registered input files)
>>> pytest -s test_system.py --recalc_all

Add --durations=0 to the pytest call to get the duration of each test; also add -vv to show all
durations (regardless of length).
"""

import argparse
import datetime
import json
import logging
import os
import re
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import pandas as pd
import pytest

import bonafide
from bonafide import AtomBondFeaturizer

####################
# Global variables #
####################

_base = os.path.dirname(__file__)

# All available data files for system testing
with open(os.path.join(_base, "system_test_data_files.json"), "r") as f:
    SYSTEM_TEST_DATA_FILES = json.load(f)

# Metadata on input strings for system testing
with open(os.path.join(_base, "input_string_metadata.json"), "r") as f:
    INPUT_STRING_METADATA = json.load(f)

# Info on features, taken from BONAFIDE
with open(os.path.join(os.path.dirname(bonafide.__file__), "_feature_info.json"), "r") as f:
    FEATURE_INFO = json.load(f)

##########################
# Load test data file(s) #
##########################


def _load_test_data(
    filter: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any], List[str], str]]:
    """Load the data from the individual system test data files (ground truth and already passed
    data).
    """
    all_tests_data = []
    n_files = 0
    for data_file, passed_file in SYSTEM_TEST_DATA_FILES:
        # Skip non-matching files if requested
        if filter is not None and data_file != filter:
            continue

        # Load data
        with open(os.path.join(_base, "data", data_file), "r") as f:
            test_data = json.load(f)
        with open(os.path.join(_base, "data", passed_file), "r") as f:
            passed = json.load(f)

        # Write data to overall test job list
        for job, job_data in test_data.items():
            all_tests_data.append(
                (data_file, job, job_data, passed, os.path.join(_base, "data", passed_file))
            )

        n_files += 1

    return all_tests_data, n_files


# Define command line argument parser for input file
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file",
    action="store",
    default=None,
    required=False,
    help="Name of the input file to be used for testing. If not provided, all available input "
    "files will be used.",
)

# Only parse known args to avoid conflicts with pytest args
args, _ = parser.parse_known_args()

# Load data
SYSTEM_TEST_DATA, n_files = _load_test_data(filter=args.input_file)
print(
    f"\n[System testing] Loaded {len(SYSTEM_TEST_DATA)} system test jobs from {n_files} datafiles."
)


####################
# Helper functions #
####################


def _fetch_feature_info(factory_name: str) -> Tuple[int, Dict[str, Any]]:
    """Read the metadata for a given feature."""
    for feature_idx, feature_info in FEATURE_INFO.items():
        if feature_info["factory"] == factory_name:
            return int(feature_idx), feature_info
    raise ValueError(f"Feature info for factory '{factory_name}' not found.")


def _fetch_input_metadata(input_string: str) -> Dict[str, Any]:
    """Read the metadata for a given input string."""
    if input_string in INPUT_STRING_METADATA:
        return INPUT_STRING_METADATA[input_string]
    raise ValueError(f"Input metadata for string '{input_string}' not found.")


def _evaluate_results(
    results_df: pd.DataFrame, name: str, feature_type: str, uses_iterable_option: bool
) -> None:
    """Assert that the results DataFrame has one column with the correct feature name and a
    correct index column name.
    """
    # Number of columns
    if len(results_df.columns) != 1 and uses_iterable_option is False:
        raise AssertionError(
            f"Results DataFrame has {len(results_df.columns)} columns instead of 1."
        )

    # Column name
    col_name = results_df.columns[0]
    if any(
        [
            re.search(r"atoms_beyond_[1-9]\d*_bonds", col_name) is not None,
            re.search(r"atoms_within_[1-9]\d*_bonds", col_name) is not None,
            re.search(r"atoms_[1-9]\d*_bonds_away", col_name) is not None,
            re.search(r"bonds_beyond_[1-9]\d*_bonds", col_name) is not None,
            re.search(r"bonds_within_[1-9]\d*_bonds", col_name) is not None,
            re.search(r"atoms_beyond_\d+\.(?:\d+)_angstrom", col_name) is not None,
            re.search(r"atoms_within_\d+\.(?:\d+)_angstrom", col_name) is not None,
            re.search(r"bonds_beyond_\d+\.(?:\d+)_angstrom", col_name) is not None,
            re.search(r"bonds_within_\d+\.(?:\d+)_angstrom", col_name) is not None,
            uses_iterable_option is True,
        ]
    ):
        pass
    elif col_name.startswith(name) is False:
        raise AssertionError(
            f"Results DataFrame column '{col_name}' does not start with expected name string "
            f"'{name}'."
        )

    # Index name
    if feature_type == "atom":
        expected_index_name = "ATOM_INDEX"
    if feature_type == "bond":
        expected_index_name = "BOND_INDEX"
    if results_df.index.name != expected_index_name:
        raise AssertionError(
            f"Results DataFrame index name '{results_df.index.name}' does not match expected "
            f"name '{expected_index_name}'."
        )


def _evaluate_results2(
    results_df: pd.DataFrame, atom_bond_indices: List[int], uses_iterable_option: bool
) -> None:
    """Assert that only for the requested atom/bond indices values were calculated."""
    l1 = results_df.dropna().index.to_list()
    l1.sort()
    l2 = atom_bond_indices.copy()
    l2.sort()
    if l1 != l2 and uses_iterable_option is False:
        raise AssertionError(
            f"Results DataFrame has values for indices {l1} instead of the expected indices {l2}."
        )


def _evaluate_results3(
    results_df: pd.DataFrame, data_type: str, expected_data: Dict[str, Any], split_str: bool
) -> None:
    """Assert the correct feature values"""
    for idx, expected_value in expected_data.items():
        value = results_df.loc[idx].values[-1]

        # Inaccessible case
        if expected_value == "_inaccessible":
            if value != "_inaccessible":
                raise AssertionError(
                    f"Feature value for atom/bond index {idx} is '{value}' instead of expected "
                    f"'{expected_value}'."
                )
            continue

        # split_str True case
        if split_str is True:
            # Edge case from Rdkit2DAtomRingInfo features
            if expected_value == "none":
                if str(value) != "none":
                    raise AssertionError(
                        f"Feature value for atom/bond index {idx} is '{value}' instead of "
                        f"expected 'none'."
                    )
                continue

            # Check number of features and actual values
            expected_values = [float(x) for x in expected_value.split(",")]
            value_values = [float(x) for x in str(value).split(",")]

            if len(expected_values) != len(value_values):
                raise AssertionError(
                    f"Feature value for atom/bond index {idx} has length "
                    f"{len(value_values)} instead of expected length {len(expected_values)}."
                )

            for v1, v2 in zip(expected_values, value_values):
                if pytest.approx(v1) != v2:
                    raise AssertionError(
                        f"Feature value for atom/bond index {idx} contains '{v2}' instead of "
                        f"expected '{v1}'."
                    )
            continue

        # Str case
        if data_type == "str":
            if expected_value != str(value):
                raise AssertionError(
                    f"Feature value for atom/bond index {idx} is '{value}' instead of expected "
                    f"'{expected_value}'."
                )
            continue

        # Int and bool case
        if data_type in ["int", "bool"]:
            if expected_value != value:
                raise AssertionError(
                    f"Feature value for atom/bond index {idx} is '{value}' instead of expected "
                    f"'{expected_value}'."
                )
            continue

        # Float case
        if data_type == "float":
            if pytest.approx(expected_value) != value:
                raise AssertionError(
                    f"Feature value for atom/bond index {idx} is '{value}' instead of expected "
                    f"'{expected_value}'."
                )
            continue

        raise RuntimeError("Something went wrong during result evaluation.")


################
# System tests #
################

START_TIME = datetime.datetime.now()
counter_new = 0
print(f"[System testing] Start time: {START_TIME}.")


@pytest.fixture(autouse=True)
def _report_end_time(request: pytest.FixtureRequest) -> Generator[Any, Any, Any]:
    yield
    id = request.node.callspec.id
    idx = int(id.split(" | ")[0])
    if idx == len(SYSTEM_TEST_DATA) - 1:
        print("\n\n[System testing] Hurray, all system tests done.")
        print(f"[System testing] Number of considered new tests:  {counter_new}.")
        END_TIME = datetime.datetime.now()
        print(f"[System testing] End time:                        {END_TIME}.")
        print(f"[System testing] Total time elapsed:              {END_TIME - START_TIME}.")


@pytest.mark.system
@pytest.mark.parametrize(
    "data_file_name, factory_name, input_string, electronic_structure_data, energy_data, "
    "configuration_settings, expected_data, split_str, passed, passed_file_path",
    [
        (
            job[0],
            job[1],
            job[2]["input_string"],
            job[2]["electronic_structure_data"],
            job[2]["energy_data"],
            job[2]["configuration_settings"],
            job[2]["expected_data"],
            job[2]["split_str"],
            job[3],
            job[4],
        )
        for job in SYSTEM_TEST_DATA
    ],
    ids=[f"{idx} | {job[1]}" for idx, job in enumerate(SYSTEM_TEST_DATA)],
)
def test_bonafide(
    request: pytest.FixtureRequest,
    caplog: pytest.LogCaptureFixture,
    fetch_data_file: Callable[[str], str],
    clean_up_logfile: Callable[[str], None],
    data_file_name: str,
    factory_name: str,
    input_string: str,
    electronic_structure_data: List[List[str]],
    energy_data: List[List[Union[int, float, str]]],
    configuration_settings: List[List[Any]],
    expected_data: Dict[str, Any],
    split_str: bool,
    passed: List[str],
    passed_file_path: str,
) -> None:
    """End-to-end system test for ``BONAFIDE``."""
    # Skip missing data cases
    if expected_data == {}:
        pytest.skip("No data for testing available yet.")

    # Print info
    print(f"\n  > Testing feature factory '{factory_name}' from '{data_file_name}'.")

    # Skip already passed tests
    if request.config.getoption("--recalc_all") is False:
        if factory_name in passed:
            print(
                "  > Test passed previously. Force retesting by adding --recalc_all to the "
                "pytest call."
            )
            return

    global counter_new
    counter_new += 1
    print(f"  > Input string: '{input_string}'.")

    # Check provided data
    if input_string is None:
        raise ValueError("'input_string' must not be None.")
    if split_str is None:
        raise ValueError("'split_str' must not be None.")

    # Fetch feature info
    feature_idx, feature_info = _fetch_feature_info(factory_name)
    print(f"  > Feature index: {feature_idx}.")

    feature_name = feature_info["name"]
    feature_type = feature_info["feature_type"]
    data_type = feature_info["data_type"]
    requires_electronic_structure_data = feature_info["requires_electronic_structure_data"]
    requires_bond_data = feature_info["requires_bond_data"]
    requires_charge = feature_info["requires_charge"]
    requires_multiplicity = feature_info["requires_multiplicity"]

    # Fetch input metadata
    input_metadata = _fetch_input_metadata(input_string)
    input_format = input_metadata["input_format"]
    charge = input_metadata["charge"]
    multiplicity = input_metadata["multiplicity"]

    # Extract atom/bond indices
    expected_data = {int(idx): value for idx, value in expected_data.items()}
    atom_bond_indices = [idx for idx in expected_data]

    # Initialize featurizer
    featurizer = AtomBondFeaturizer()

    # Read input
    if input_format == "file":
        input_string = fetch_data_file(file_name=input_string)
    featurizer.read_input(
        input_value=input_string, namespace="irrelevant", input_format=input_format
    )

    # Define bonds if required
    if requires_bond_data is True and input_string.endswith(".xyz"):
        featurizer.determine_bonds()

    # Attach electronic structure data
    if requires_electronic_structure_data is True:
        if len(electronic_structure_data) == 0:
            raise ValueError(
                f"Feature '{factory_name}' requires electronic structure data, but none was "
                "provided."
            )
        for entry in electronic_structure_data:
            _file_path = fetch_data_file(file_name=entry[0])
            featurizer.attach_electronic_structure(
                electronic_structure_data=_file_path, state=entry[1]
            )

    # Attach energy data
    if len(energy_data) != 0:
        for entry in energy_data:
            energy_data = tuple(entry[0])
            state = entry[1]
            featurizer.attach_energy(energy_data=energy_data, state=state)

    # Add charge and multiplicity
    if requires_charge is True:
        featurizer.set_charge(charge=charge)
    if requires_multiplicity is True:
        featurizer.set_multiplicity(multiplicity=multiplicity)

    # Change configuration settings if required
    if len(configuration_settings) > 0:
        configuration_settings = [tuple(setting) for setting in configuration_settings]
        featurizer.set_options(configs=configuration_settings)

    # Check molecule vault
    assert all(featurizer.mol_vault.is_valid)

    # Calculate features
    if feature_type == "atom":
        featurizer.featurize_atoms(atom_indices=atom_bond_indices, feature_indices=feature_idx)
    if feature_type == "bond":
        featurizer.featurize_bonds(bond_indices=atom_bond_indices, feature_indices=feature_idx)

    # Determine if feature is based on an iterable option (necessary to know for the evaluation
    # of the shape of the results DataFrame)
    _uses_iterable_option = False
    for record in caplog.records:
        if all(
            [
                record.levelno == logging.WARNING,
                "The 'atom_indices' parameter was set to 'all' because at least one of the "
                "selected features requires its iterable option base feature(s) to be available "
                "for all atoms." in record.msg,
            ]
        ):
            _uses_iterable_option = True

    # Get data and check results
    if feature_type == "atom":
        results_df = featurizer.return_atom_features()
    if feature_type == "bond":
        results_df = featurizer.return_bond_features()

    _evaluate_results(
        results_df=results_df,
        name=feature_name,
        feature_type=feature_type,
        uses_iterable_option=_uses_iterable_option,
    )
    _evaluate_results2(
        results_df=results_df,
        atom_bond_indices=atom_bond_indices,
        uses_iterable_option=_uses_iterable_option,
    )
    _evaluate_results3(
        results_df=results_df, data_type=data_type, expected_data=expected_data, split_str=split_str
    )
    print("  > Results evaluation passed.")

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno != logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()

    # Mark test as passed
    if factory_name not in passed:
        passed.append(factory_name)
        passed.sort()
        with open(passed_file_path, "w") as f:
            json.dump(passed, f, indent=4)
    print(f"  > Updated '{os.path.split(passed_file_path)[-1]}'.")
    print("  > All done.")
