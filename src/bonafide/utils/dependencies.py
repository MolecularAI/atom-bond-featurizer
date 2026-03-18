"""Utility module to check for required dependencies that are accessed through a Python
subprocess.
"""

import logging
import os
import shutil
import tempfile
from subprocess import run
from typing import List

from bonafide.utils.constants import XTB_VERSION_STRING
from bonafide.utils.helper_functions import get_function_or_method_name


def _check_xtb_version() -> bool:
    """Check if the correct xtb version is installed.

    Returns
    -------
    bool
        ``True`` if the correct xtb version is installed, ``False`` otherwise.
    """
    try:
        res = run(["xtb", "--version"], check=False, capture_output=True, text=True)
    except Exception:
        return False

    if XTB_VERSION_STRING not in res.stdout.strip():
        return False
    return True


def check_dependency_path(prg_name: str) -> str:
    """Check if a required program is installed and accessible in the system PATH.

    Parameters
    ----------
    prg_name : str
        The name of the program to check for.

    Returns
    -------
    str
        The path to the program if it is found.
    """
    _loc = get_function_or_method_name()

    path = shutil.which(prg_name)
    if path is None:
        _errmsg = (
            f"Required program '{prg_name}' is not installed or not found in PATH. "
            "Install it and/or add it to your PATH environment variable."
        )
        raise ImportError(f"{_loc}(): {_errmsg}")

    # Check xtb version (important because Fukui indices are not correctly printed in older
    # versions)
    if prg_name == "xtb":
        if _check_xtb_version() is False:
            _errmsg = "Installed xtb version is not supported. Install xtb version 6.7.1."
            raise ImportError(f"{_loc}(): {_errmsg}")

    return path


def check_dependency_env(python_path: str, package_names: List[str], namespace: str) -> str:
    """Check if a required package is installed in a given Python environment.

    It is first checked if the provided Python interpreter path is valid. Then, a temporary
    Python script is created that checks if the required package is installed in the external
    environment.

    Parameters
    ----------
    python_path : str
        The path to the Python interpreter where the package is expected to be installed.
    package_names : List[str]
        A list of the package to check for.
    namespace : str
        The namespace of the currently handled molecule for logging purposes.

    Returns
    -------
    str
        The path to the Python interpreter if the package is found.
    """
    _loc = get_function_or_method_name()

    # Check if provided path is a valid Python interpreter
    python_path = os.path.expanduser(python_path)
    if not os.path.exists(python_path):
        _errmsg = f"Provided Python interpreter path '{python_path}' is not valid."
        logging.error(f"'{namespace}' | {_loc}()\n{_errmsg}")
        raise ImportError(f"{_loc}(): {_errmsg}")

    # Check if the required package is installed
    for package_name in package_names:
        # Check script
        check_script = [
            "try:",
            "    from importlib.metadata import distributions",
            "except ImportError:",
            "    from importlib_metadata import distributions",
            'packages = [dist.metadata["Name"] for dist in distributions()]',
            f"if '{package_name}' not in packages:",
            "    print(False)",
            "else:",
            "    print(True)",
        ]
        check_script_str = "\n".join(check_script)

        # Write the script to a temporary file
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(check_script_str)
            tmp = f.name

        # Run script
        res = run([python_path, tmp], check=False, capture_output=True, text=True)

        # Check result
        if res.stdout.strip() != "True":
            _errmsg = (
                f"Required package '{package_name}' is not installed in the "
                f"environment of '{python_path}'."
            )
            logging.error(f"'{namespace}' | {_loc}()\n{_errmsg}")
            raise ImportError(f"{_loc}(): {_errmsg}")

    return python_path
