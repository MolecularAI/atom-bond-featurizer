"""Helper functions for the featurization examples."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd


def read_csv(input_file_path: str) -> pd.DataFrame:
    """Read a CSV file and return a pandas DataFrame.

    Parameters
    ----------
    input_file_path : str
        The path to the input CSV file.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data from the input CSV file.
    """
    df = pd.read_csv(input_file_path)
    return df


def read_energies_files(
    input_file_path: str,
) -> Tuple[
    Tuple[Optional[float], Optional[str]],
    Tuple[Optional[float], Optional[str]],
    Tuple[Optional[float], Optional[str]],
]:
    """Read energies and units from a text file and return them as tuples.

    Parameters
    ----------
    input_file_path : str
        The path to the input text file containing the energies and units.

    Returns
    -------
    Tuple[Tuple[float, str], Tuple[float, str], Tuple[float, str]]
        A tuple containing three tuples: (energy_n, unit_n), (energy_nplus1, unit_nplus1), and
        (energy_nminus1, unit_nminus1). Each inner tuple contains a float representing the energy
        and a string representing the unit.
    """
    with open(input_file_path, "r") as f:
        lines = f.readlines()

    energy_n = None
    energy_nplus1 = None
    energy_nminus1 = None

    unit_n = None
    unit_nplus1 = None
    unit_nminus1 = None

    for line in lines:
        splitted = line.strip().split()
        if "Energy (n)" in line:
            energy_n = float(splitted[-2])
            unit_n = splitted[-1]

        if "Energy (n+1)" in line:
            energy_nplus1 = float(line.split()[-2])
            unit_nplus1 = splitted[-1]

        if "Energy (n-1)" in line:
            energy_nminus1 = float(line.split()[-2])
            unit_nminus1 = splitted[-1]

    return (energy_n, unit_n), (energy_nplus1, unit_nplus1), (energy_nminus1, unit_nminus1)


def parallelizer(
    worker: Callable[[Any], Tuple[str, Any]], input_list: List[Any], n_processors: int
) -> Dict[str, Any]:
    """Parallelize the execution of a worker function over an input list.

    Parameters
    ----------
    worker : Callable
        The worker function to be executed in parallel. It should take a single input and return a tuple of (name, result).
    input_list : List[Any]
        A list of inputs to be processed by the worker function.
    n_processors : int
        The number of processors to use for parallel execution.

    Returns
    -------
    Dict[str, Any]
        A dictionary mapping the name returned by the worker function to its corresponding result.
    """
    all_results = {}
    with ProcessPoolExecutor(max_workers=n_processors) as executor:
        tasks = {executor.submit(worker, inp): inp for inp in input_list}

        for future in as_completed(tasks):
            name, res = future.result()
            all_results[name] = res

    return all_results
