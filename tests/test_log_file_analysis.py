from typing import Callable, Optional

import pandas as pd
import pytest

from bonafide import LogFileAnalyzer


@pytest.mark.log_file_analysis
@pytest.mark.parametrize("log_level", [None, "ERROR", "warniNG", "debug"])
def test_get_level_log_messages(
    fetch_data_file: Callable[[str], str],
    log_level: Optional[str],
) -> None:
    """Test for the ``get_level_log_messages()`` method."""
    a = LogFileAnalyzer(fetch_data_file("bonafide_example_log.log"))

    if log_level is None:
        res = a.get_level_log_messages()
    else:
        res = a.get_level_log_messages(log_level=log_level)

    assert type(res) == str

    if log_level is None or log_level.upper() == "ERROR":
        assert res.count("| ERROR |") == 2
        assert (
            "The initialized molecule vault is of dimensionality '2D'. All bonds are already "
            "defined."
        ) in res
        assert (
            "The initialized molecule vault is of dimensionality '2D'. Electronic structure "
            "calculations are not feasible for 2D"
        ) in res
    elif log_level.upper() == "WARNING":
        assert res.count("| WARNING |") == 1
        assert (
            "The input to 'output_directory' is None (default value). Therefore, all output files "
            "potentially generated "
        ) in res
    elif log_level.upper() == "DEBUG":
        assert res == ""


@pytest.mark.log_file_analysis
@pytest.mark.parametrize(
    "target_string, expected_result",
    [("Please provide a 3D ensemble.", True), ("I will not be found in the last line.", False)],
)
def test_check_string_in_last_line(
    fetch_data_file: Callable[[str], str], target_string: str, expected_result: bool
) -> None:
    """Test for the ``check_string_in_last_line()`` method."""
    a = LogFileAnalyzer(fetch_data_file("bonafide_example_log.log"))
    res = a.check_string_in_last_line(target_string=target_string)
    assert type(res) == bool
    assert res == expected_result


@pytest.mark.log_file_analysis
def test_get_time_for_individual_features(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``get_time_for_individual_features()`` method."""
    a = LogFileAnalyzer(fetch_data_file("bonafide_example_log2.log"))
    res = a.get_time_for_individual_features()

    assert type(res) == pd.DataFrame
    assert res.shape == (4, 4)
    assert res.columns.tolist() == ["elapsed_time [s]", "start_time", "end_time", "feature_type"]

    assert res.loc["alfabet2D-bond-bond_dissociation_energy"]["elapsed_time [s]"] == 8
    assert res.loc["qmdesc2D-atom-fukui_minus"]["elapsed_time [s]"] == 0
    assert res.loc["alfabet2D-bond-bond_dissociation_free_energy"]["elapsed_time [s]"] == 0
    assert res.loc["mendeleev2D-atom-en_sanderson"]["elapsed_time [s]"] == 1


@pytest.mark.log_file_analysis
def test_get_total_time_for_atom_featurization(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``get_total_time_for_atom_featurization()`` method."""
    a = LogFileAnalyzer(fetch_data_file("bonafide_example_log2.log"))
    res = a.get_total_time_for_atom_featurization()
    assert type(res) == float
    assert res == 1.0


@pytest.mark.log_file_analysis
def test_get_total_time_for_bond_featurization(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``get_total_time_for_bond_featurization()`` method."""
    a = LogFileAnalyzer(fetch_data_file("bonafide_example_log2.log"))
    res = a.get_total_time_for_bond_featurization()
    assert type(res) == float
    assert res == 8.0


@pytest.mark.log_file_analysis
def test_get_total_runtime(fetch_data_file: Callable[[str], str]) -> None:
    """Test for the ``get_total_runtime()`` method."""
    a = LogFileAnalyzer(fetch_data_file("bonafide_example_log.log"))
    res = a.get_total_runtime()
    assert type(res) == float
    assert res == 28.0
