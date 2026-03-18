"""Test functions for the ``bonafide.utils.dependencies`` module."""

import os
import sys

import pytest

from bonafide.utils.dependencies import check_dependency_env, check_dependency_path

###################################################
# Tests for the check_dependency_path() function. #
###################################################


@pytest.mark.check_dependency_path
def test_check_dependency_path() -> None:
    """Test for the ``check_dependency_path()`` function: valid path."""
    _cmd = "python"
    path = check_dependency_path(prg_name=_cmd)
    assert path is not None
    assert path.endswith(_cmd)


@pytest.mark.check_dependency_path
def test_check_dependency_path2() -> None:
    """Test for the ``check_dependency_path()`` function: invalid path."""
    _cmd = "this_program_does_really_not_exist_for_sure"
    with pytest.raises(
        ImportError, match=f"Required program '{_cmd}' is not installed or not found in PATH."
    ):
        check_dependency_path(prg_name=_cmd)


##################################################
# Tests for the check_dependency_env() function. #
##################################################


@pytest.mark.check_dependency_env
def test_check_dependency_env() -> None:
    """Test for the ``check_dependency_env()`` function: valid packages."""
    python_path = check_dependency_env(
        python_path=sys.executable,
        package_names=["bonafide", "pandas", "numpy"],
        namespace="irrelevant",
    )
    assert python_path == sys.executable


@pytest.mark.check_dependency_env
def test_check_dependency_env2() -> None:
    """Test for the ``check_dependency_env()`` function: invalid python path."""
    _python_path = os.path.join(
        "this", "Python", "does", "really", "not", "exist", "for", "sure", "bin", "python"
    )
    with pytest.raises(
        ImportError, match=f"Provided Python interpreter path '{_python_path}' is not valid."
    ):
        check_dependency_env(
            python_path=_python_path,
            package_names=["bonafide", "pandas", "numpy"],
            namespace="irrelevant",
        )


@pytest.mark.check_dependency_env
def test_check_dependency_env3() -> None:
    """Test for the ``check_dependency_env()`` function: missing package."""
    _python_path = sys.executable
    _missing_package = "this_package_does_really_not_exist_for_sure"

    with pytest.raises(
        ImportError,
        match=f"Required package '{_missing_package}' is not installed in "
        f"the environment of '{_python_path}'.",
    ):
        check_dependency_env(
            python_path=_python_path,
            package_names=[
                "bonafide",
                "pandas",
                "numpy",
                _missing_package,
            ],
            namespace="irrelevant",
        )
