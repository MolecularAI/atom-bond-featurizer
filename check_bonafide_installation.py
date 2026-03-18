"""Check BONAFIDE installation."""

import importlib
import os
import shutil
import tomllib
import warnings

warnings.filterwarnings(action="ignore")


def _check_module(module_name: str) -> bool:
    """Check if a module can be imported.

    Parameters
    ----------
    module_name : str
        The name of the module.

    Returns
    -------
    bool
        ``True`` if the module can be imported, ``False`` otherwise.
    """
    try:
        importlib.import_module(name=module_name)

    except ImportError:
        print("not installed.")
        return False

    print("installed.")
    return True


def _check_exe(prg_name: str) -> bool:
    """Check if an executable is available in PATH.

    Parameters
    ----------
    prg_name : str
        The name of the program.

    Returns
    -------
    bool
        ``True`` if the executable is found, ``False`` otherwise.
    """
    prg = shutil.which(cmd=prg_name)
    if prg is None:
        print("not installed.")
        return False

    if os.access(path=prg, mode=os.X_OK) is False:
        print("accessible but not executable.")
        return False

    print("installed.")
    return True


def _check_qmdesc() -> bool:
    """Check qmdesc installation.

    Returns
    -------
    bool
        ``True`` if qmdesc is correctly installed, ``False`` otherwise.
    """
    try:
        from qmdesc import ReactivityDescriptorHandler

    except ImportError:
        print("not installed.")
        return False

    try:
        handler = ReactivityDescriptorHandler()
        _ = handler.predict("CCO")

    except Exception as e:
        print(f"not installed correctly: {e}")
        return False

    print("installed.")
    return True


def _check_alfabet() -> bool:
    """Check alfabet installation.

    Returns
    -------
    bool
        ``True`` if alfabet is correctly installed, ``False`` otherwise.
    """
    try:
        import bonafide
    except ImportError:
        print("installation cannot be checked (bonafide module not found).")
        return False

    config_path = os.path.join(os.path.dirname(bonafide.__file__), "_feature_config.toml")

    with open(config_path, "rb") as config_file:
        configs = tomllib.load(config_file)

    interpreter_path = configs.get("alfabet", {}).get("python_interpreter_path", None)
    if interpreter_path is None:
        print(
            "not installed (Python interpreter not found in '_feature_config.toml' "
            "(alfabet.python_interpreter_path))."
        )
        return False

    interpreter_path = os.path.expanduser(path=interpreter_path)
    if os.path.isfile(interpreter_path) is False:
        print(f"not installed (Python interpreter path is invalid, {interpreter_path}).")
        return False

    command = f'"{interpreter_path}" -c "import alfabet"'
    exit_code = os.system(command=command)
    if exit_code != 0:
        print("not installed.")
        return False

    print(f"installed (in external environment at {interpreter_path}).")
    return True


def main() -> bool:
    """Run all checks and print the results.

    Returns
    -------
    bool
        ``True`` if all checks pass, ``False`` otherwise.
    """
    checks = [
        "morfeus",
        "mendeleev",
        "rdkit",
        "qmdesc",
        "kallisto",
        "psi4",
        "dftd3",
        "dbstep",
        "dscribe",
        "Multiwfn_noGUI",
        "xtb",
        "bonafide",
    ]
    checks = sorted(checks, key=str.lower)
    checks.append("alfabet")

    counter = 0
    for check in checks:
        print(f"> {check:17} ...   ", end=" ")
        if check not in ["Multiwfn_noGUI", "xtb", "qmdesc", "alfabet"]:
            if _check_module(module_name=check) is True:
                counter += 1

        elif check == "Multiwfn_noGUI" or check == "xtb":
            if _check_exe(prg_name=check) is True:
                counter += 1

        elif check == "qmdesc":
            if _check_qmdesc() is True:
                counter += 1

        elif check == "alfabet":
            if _check_alfabet() is True:
                counter += 1

        else:
            print("Check not defined.")

    if counter == len(checks):
        return True
    else:
        return False


if __name__ == "__main__":
    print("--------------------------")
    print("BONAFIDE dependency checks")
    print("--------------------------")
    if main() is True:
        print("\n==> Hurray, all dependencies are installed correctly!\n")
    else:
        print("\n==> Caution, some dependencies are missing or not correctly installed.\n")
