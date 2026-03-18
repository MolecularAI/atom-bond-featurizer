"""General helper functions for small common tasks."""

import glob
import inspect
import os
from typing import Any, Dict, List, Optional, Tuple


def flatten_dict(dictionary: Dict[str, Any], all_keys: List[str]) -> List[str]:
    """Flatten a nested dictionary and return a list of all keys.

    The input dictionary is recursively traversed, and all keys are collected. The keys are
    converted to lowercase to ensure uniformity.

    Parameters
    ----------
    dictionary : Dict[str, Any]
        The dictionary to be flattened.
    all_keys : List[str]
        A list to store all keys found in the dictionary.

    Returns
    -------
    List[str]
        A list of all keys in the dictionary.
    """
    all_keys.extend([str(k).lower() for k in list(dictionary.keys())])
    for value in dictionary.values():
        if isinstance(value, dict):
            all_keys = flatten_dict(value, all_keys)
    return all_keys


def clean_up(to_be_removed: List[str]) -> None:
    """Remove temporary files that should not be kept within the current working directory.

    All files that match the patterns specified are deleted.

    Parameters
    ----------
    to_be_removed : List[str]
        A list of glob patterns that match the files to be removed.

    Returns
    -------
    None
    """
    for pattern in to_be_removed:
        for file in glob.glob(pattern):
            if os.path.isfile(file):
                os.remove(file)


def standardize_string(inp_data: Any, case: str = "lower") -> str:
    """Standardize a string by removing leading and trailing whitespace and converting it to
    lowercase or uppercase.

    Parameters
    ----------
    inp_data : Any
        The input data to be standardized.
    case : str, optional
        The case to convert the string to, either "lower" or "upper", by default "lower".

    Returns
    -------
    str
        The standardized string.
    """
    if case == "lower":
        return str(inp_data).strip().lower()
    if case == "upper":
        return str(inp_data).strip().upper()
    return str(inp_data).strip()


def matrix_parser(
    files_lines: List[str], n_atoms: int
) -> Tuple[Optional[List[List[float]]], Optional[str]]:
    """Parse a 2D matrix from the lines of a file.

    The matrix must be in this format:

    .. code-block:: text

            1    2    3    4
        1  0.1  0.2  0.3  0.4
        2  0.5  0.6  0.7  0.8
        3  0.9  1.0  1.1  1.2
        4  1.3  1.4  1.5  1.6
        5  1.7  1.8  1.9  2.0
        6  2.1  2.2  2.3  2.4
            5   6
        1 2.5 2.6
        2 2.7 2.8
        3 2.9 3.0
        4 3.1 3.2
        5 3.3 3.4
        6 3.5 3.6

    An error message is returned if the parsing fails or the number of elements per row is
    inconsistent.

    Parameters
    ----------
    files_lines : List[str]
        The respective lines of the file with the matrix data.
    n_atoms : int
        The number of atoms in the molecule.

    Returns
    -------
    Tuple[Optional[List[List[float]]], Optional[str]]
        A tuple containing:

        * the parsed matrix as a list of lists of floats, or ``None`` if an error
          occurred, and
        * an error message if applicable (``None`` if no error occurred).
    """
    matrix_block: List[List[float]] = [[] for _ in range(n_atoms)]
    _errmsg = None
    counter = 0

    # Read the matrix elements
    try:
        for line in files_lines[1:]:
            if line.strip() == "":
                break
            if "." not in line:
                counter = 0
                continue
            matrix_block[counter].extend([float(x) for x in line.split()][1:])
            counter += 1
    except Exception as e:
        _errmsg = f"error while parsing the 2D matrix: {e}"
        return None, _errmsg

    # Check the parsed data
    for row in matrix_block:
        if len(row) != n_atoms:
            _errmsg = "error while parsing the 2D matrix: inconsistent number of elements per row."
            return None, _errmsg

    return matrix_block, _errmsg


def get_function_or_method_name() -> str:
    """Get the name of the calling function or method.

    Returns
    -------
    str
        The name of the calling function or method, or "unknown_function_or_method" if unavailable.
    """
    frame = inspect.currentframe()
    if frame is None:
        return "unknown_function_or_method"

    caller_frame = frame.f_back
    if caller_frame is None:
        return "unknown_function_or_method"

    return caller_frame.f_code.co_name
