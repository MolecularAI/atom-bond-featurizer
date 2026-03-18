"""Global fixtures."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Callable, Generator, List, Union

import pytest

from bonafide.bonafide import AtomBondFeaturizer
from bonafide.utils.molecule_vault import MolVault


@pytest.fixture(autouse=True)
def _set_pytest_testing_env_var() -> Generator[None, None, None]:
    """Set an environment variable to indicate a pytest testing session."""
    os.environ["BONAFIDE_PYTEST_TESTING_SESSION"] = "1"
    yield
    os.environ.pop(key="BONAFIDE_PYTEST_TESTING_SESSION", default=None)


@pytest.fixture(autouse=True)
def _check_tests_dir_content() -> Generator[None, None, None]:
    """Ensure tests directory content is not modified by tests."""
    initial_dir_contents = os.listdir()
    yield

    # Exclude bonafide.log and irrelevant_out_dir
    if os.path.isfile("bonafide.log") is True:
        os.remove("bonafide.log")
    if os.path.isdir("irrelevant_out_dir") is True:
        shutil.rmtree("irrelevant_out_dir")

    final_dir_contents = os.listdir()
    assert initial_dir_contents == final_dir_contents, (
        "Tests directory was modified by a test. This cannot happen!"
    )


@pytest.fixture(autouse=True)
def _set_module_log_level(caplog: pytest.LogCaptureFixture) -> Generator[None, None, None]:
    """Log level setting."""
    caplog.clear()
    caplog.set_level(logging.DEBUG)
    yield


@pytest.fixture
def fetch_data_file() -> Callable[[str], str]:
    """File path formatting."""

    def _get_path(file_name: str) -> str:
        try:
            _path = os.path.join(os.path.dirname(__file__), "data", file_name)
        except Exception:
            _path = "None"
        if os.path.exists(_path):
            return _path
        raise FileNotFoundError(f"Test data file at {_path} not found.")

    return _get_path


@pytest.fixture
def fresh_mol_vault() -> Callable[[List[str], str, str], MolVault]:
    """Provide a fresh instance of the MolVault class."""

    def _setup(mol_inputs: List[str], namespace: str, input_type: str) -> MolVault:
        return MolVault(
            mol_inputs=mol_inputs,
            namespace=namespace,
            input_type=input_type,
        )

    return _setup


@pytest.fixture
def fresh_featurizer() -> Callable[[str], AtomBondFeaturizer]:
    """Provide a fresh instance of the AtomBondFeaturizer class."""

    def _setup(log_file_name: str = "_use_default_log") -> AtomBondFeaturizer:
        if log_file_name == "_use_default_log":
            return AtomBondFeaturizer()
        return AtomBondFeaturizer(
            log_file_name=log_file_name,
        )

    return _setup


@pytest.fixture
def check_dependency() -> Callable[[str], None]:
    """Check if a dependency is installed."""

    def _setup(executable_name: str) -> None:
        if shutil.which(executable_name) is None:
            raise EnvironmentError(
                f"{executable_name} executable not found. Please install {executable_name} to run "
                "this test."
            )

    return _setup


@pytest.fixture
def clean_up_logfile() -> Callable[[str], None]:
    """Remove the log file after a test."""

    def _setup(file_path: str = "_use_default_log") -> None:
        if file_path == "_use_default_log":
            file_path = "bonafide.log"
        try:
            os.remove(file_path)
        except Exception:
            pass

    return _setup


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add a command line option for:

    * specifying the name of the input file. If not provided, all available input files will be
    used. This is only relevant for the system tests.

    * requesting the recalculation of all feature values and the comparison with the expected
    values. This is only relevant for the system tests.
    """
    parser.addoption(
        "--input_file",
        action="store",
        default=None,
        help="Name of the input file to be used for testing. If not provided, all available input "
        "files will be used.",
    )

    parser.addoption(
        "--recalc_all",
        action="store_true",
        help="Recalculate all features and compare them to the expected values.",
    )


@pytest.fixture
def recalc_all(request: pytest.FixtureRequest) -> bool:
    """Fixture for the ``--recalc_all`` command line option for system testing."""
    return request.config.getoption("--recalc_all")


@pytest.fixture
def input_file(request: pytest.FixtureRequest) -> Union[str, None]:
    """Fixture for the ``--input_file`` command line option for system testing."""
    return request.config.getoption("--input_file")
