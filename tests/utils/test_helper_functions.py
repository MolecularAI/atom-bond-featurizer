"""Test functions for the ``bonafide.utils.helper_functions`` module."""

import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pytest
from pytest_mock import MockerFixture

from bonafide.utils.helper_functions import (
    clean_up,
    flatten_dict,
    get_function_or_method_name,
    matrix_parser,
    standardize_string,
)

##########################################
# Tests for the flatten_dict() function. #
##########################################


@pytest.mark.flatten_dict
@pytest.mark.parametrize(
    "input_dict, empty_start, all_expected_keys",
    [
        ({"Key1": 10, "key2": 20, "KEY3": 30}, [], ["key1", "key2", "key3"]),
        ({"key1": 1}, [], ["key1"]),
        ({}, [], []),
        (
            {
                "Key1": 10,
                "key2": {"subkey2a": 20, "subkey2b": 21, "subkey2c": {"subsubkey2": 22}},
                "KEY3": 30,
            },
            [],
            [
                "key1",
                "key2",
                "subkey2a",
                "subkey2b",
                "subkey2c",
                "subsubkey2",
                "key3",
            ],
        ),
        (
            {
                "Key1": 10,
                "key2": {"subkey2a": 20, "subkey2b": 21, "subkey2c": {"subsubkey2": 22}},
                "KEY3": 30,
            },
            ["initial_key"],
            [
                "initial_key",
                "key1",
                "key2",
                "subkey2a",
                "subkey2b",
                "subkey2c",
                "subsubkey2",
                "key3",
            ],
        ),
        (
            {"KEY1": {"subkey1": {"subsubkey1": {"subSUBsubkey1": 1}}}},
            [],
            ["key1", "subkey1", "subsubkey1", "subsubsubkey1"],
        ),
    ],
)
def test_flatten_dict(input_dict: Dict, empty_start: List, all_expected_keys: List) -> None:
    """Test for the ``flatten_dict()`` function."""
    all_keys = flatten_dict(dictionary=input_dict, all_keys=empty_start)
    assert type(all_keys) == list
    assert all(isinstance(k, str) for k in all_keys)
    assert sorted(all_keys) == sorted(all_expected_keys)


######################################
# Tests for the clean_up() function. #
######################################


@pytest.mark.clean_up
def test_clean_up() -> None:
    """Test for the ``clean_up()`` function."""
    # Create temporary directory for testing
    temp_dir = "_temp_test_dir"
    os.mkdir(temp_dir)

    # Create temporary files
    temp_files = [
        "charges",
        "temp_test_file1.tmp",
        "temp_test_file2.tmp",
        "temp_test_file2a.tmpx",
        "not_a_temp_file.txt",
    ]
    temp_file_paths = [os.path.join(temp_dir, file) for file in temp_files]
    for path in temp_file_paths:
        with open(path, "w") as f:
            f.write("Temporary file content\n")

    # Ensure files are created
    assert len(os.listdir(temp_dir)) == len(temp_files)
    for path in temp_file_paths:
        assert os.path.isfile(path)

    # Clean up
    os.chdir(temp_dir)
    patterns_to_remove = ["*.tmp", "*.tmpx", "charges"]
    clean_up(to_be_removed=patterns_to_remove)
    os.chdir("..")

    # Check that files are removed
    assert len(os.listdir(temp_dir)) == 1
    assert not os.path.isfile(temp_file_paths[0])
    assert not os.path.isfile(temp_file_paths[1])
    assert not os.path.isfile(temp_file_paths[2])
    assert not os.path.isfile(temp_file_paths[3])
    assert os.path.isfile(temp_file_paths[4])

    # Clean up
    shutil.rmtree(temp_dir)


################################################
# Tests for the standardize_string() function. #
################################################


@pytest.mark.standardize_string
@pytest.mark.parametrize(
    "input_data, case_option, expected_output",
    [
        ("This is a test", "lower", "this is a test"),
        ("This is another test", "upper", "THIS IS ANOTHER TEST"),
        ("   WHAT   happens HERE?", "lower", "what   happens here?"),
        ("", "lower", ""),
        ("2d", "upper", "2D"),
        ("   \n", "upper", ""),
        (12345, "lower", "12345"),
        (None, "upper", "NONE"),
        ("15% OF $100", "lower", "15% of $100"),
        ("Test default CASE.\n", None, "test default case."),
    ],
)
def test_standardize_string(input_data: Any, case_option: str, expected_output: str) -> None:
    """Test for the ``standardize_string()`` function."""
    if case_option is None:
        assert standardize_string(inp_data=input_data) == expected_output
    else:
        assert standardize_string(inp_data=input_data, case=case_option) == expected_output


###########################################
# Tests for the matrix_parser() function. #
###########################################


@pytest.mark.matrix_parser
@pytest.mark.parametrize(
    "input_file, n_atoms, _expected_data, _expected_error_msg",
    [
        (
            "2D_matrix.txt",
            6,
            [((0, 0), 0.1), ((4, 0), 1.7), ((5, 5), 3.6), ((5, 3), 2.4), ((0, 4), 2.5)],
            None,
        ),
        (
            "2D_matrix_broken.txt",
            6,
            None,
            "error while parsing the 2D matrix: inconsistent number of elements per row.",
        ),
        (
            "2D_matrix_broken2.txt",
            6,
            None,
            "error while parsing the 2D matrix: could not convert string to float",
        ),
        (
            "2D_matrix_broken3.txt",
            6,
            None,
            "error while parsing the 2D matrix: inconsistent number of elements per row.",
        ),
    ],
)
def test_matrix_parser(
    fetch_data_file: Callable[[str], str],
    input_file: str,
    n_atoms: int,
    _expected_data: Union[List[Union[Tuple[int, int], float]], None],
    _expected_error_msg: Optional[str],
) -> None:
    """Test for the ``matrix_parser()`` function."""
    # Read input file
    input_path = fetch_data_file(file_name=input_file)
    with open(input_path, "r") as f:
        input_lines = f.readlines()

    # Parse matrix
    parsed_matrix, error_message = matrix_parser(files_lines=input_lines, n_atoms=n_atoms)

    # Check results
    if _expected_error_msg is None:
        assert error_message is None
        assert parsed_matrix is not None
        assert type(parsed_matrix) == list
        assert len(parsed_matrix) == n_atoms
        assert all([type(row) == list for row in parsed_matrix])
        assert all([len(row) == n_atoms for row in parsed_matrix])

        for pos, val in _expected_data:
            i, j = pos
            assert pytest.approx(parsed_matrix[i][j]) == val

    else:
        assert parsed_matrix is None
        assert type(error_message) == str
        assert _expected_error_msg in error_message


#########################################################
# Tests for the get_function_or_method_name() function. #
#########################################################


def _test_function():
    loc = get_function_or_method_name()
    return loc


class _TestClass:
    def _test_method(self):
        loc = get_function_or_method_name()
        return loc


@pytest.mark.get_function_or_method_name
@pytest.mark.parametrize(
    "tester, expected_name",
    [
        (_test_function, "_test_function"),
        (_TestClass()._test_method, "_test_method"),
    ],
)
def test_get_function_or_method_name(tester: Callable[[], str], expected_name: str) -> None:
    """Test for the ``get_function_or_method_name()`` function: normal cases."""
    assert tester() == expected_name


@pytest.mark.get_function_or_method_name
def test_get_function_or_method_name2(
    mocker: MockerFixture,
) -> None:
    """Test for the ``get_function_or_method_name()`` function: edge cases."""
    # Test when currentframe returns None
    mock = mocker.patch("bonafide.utils.helper_functions.inspect.currentframe", return_value=None)
    assert _test_function() == "unknown_function_or_method"

    t = _TestClass()
    assert t._test_method() == "unknown_function_or_method"

    assert mock.call_count == 2

    # Test when caller_frame is None
    mock_frame = mocker.MagicMock()
    mock_frame.f_back = None
    mock = mocker.patch(
        "bonafide.utils.helper_functions.inspect.currentframe", return_value=mock_frame
    )
    assert _test_function() == "unknown_function_or_method"

    t = _TestClass()
    assert t._test_method() == "unknown_function_or_method"

    assert mock.call_count == 2
