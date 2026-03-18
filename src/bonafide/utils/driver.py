"""Drivers for xtb, Multiwfn, kallisto, and any other external programs."""

import logging
import os
import shutil
import tempfile
from subprocess import PIPE, CompletedProcess, run
from typing import Any, Dict, List, Optional, Tuple, Union

from bonafide.utils.dependencies import check_dependency_env, check_dependency_path
from bonafide.utils.environment import Environment
from bonafide.utils.helper_functions import clean_up, get_function_or_method_name


def xtb_driver(
    input_dict: Dict[str, Optional[Union[int, float, str]]],
    environment_variables: Dict[str, Optional[str]],
) -> Tuple[int, str]:
    """Run ``xtb`` with the provided input parameters and environment variables.

    The xtb command is constructed based on the input dictionary, and the environment variables
    are set before running xtb. After the run, the environment is reset.

    Parameters
    ----------
    input_dict : Dict[str, Optional[Union[int, float, str]]]
        A dictionary containing the input parameters for xtb. It should include:

        * "input_file_path": Path to the input file for xtb.
        * "output_file_path": Path to save the output of xtb.
        * Other xtb options as key-value pairs.
    environment_variables : Dict[str, Optional[str]]
        A dictionary containing the environment variables to set before running xtb with the
        respective values.

    Returns
    -------
    Tuple[int, str]
        A tuple containing the return code of the xtb command and any error message produced
        during execution.
    """
    # Check if xtb program is available
    _call = "xtb"
    _ = check_dependency_path(prg_name=_call)

    # Build xtb command
    xtb_command = [_call, str(input_dict["input_file_path"])]
    for option, argument in input_dict.items():
        if option in ["input_file_path", "output_file_path"]:
            continue
        xtb_command.append(f"--{option}")
        if argument is not None:
            xtb_command.append(str(argument))

    # Handle environment variables
    env = Environment(**environment_variables)
    env.set_environment()

    # Run xtb
    res = run(xtb_command, stdout=PIPE, stderr=PIPE, check=False)
    return_code = res.returncode
    stderr = res.stderr.decode()

    # Write output file
    with open(str(input_dict["output_file_path"]), "wb") as f:
        f.write(res.stdout)

    # Reset environment
    env.reset_environment()

    return return_code, stderr


def _modify_settings_ini(nprocs: int, modify_ispecial: bool) -> None:
    """Modify the Multiwfn-specific settings file (settings.ini) to set the number of threads.
    Additionally, the "ispecial" setting can be set to 1 if requested by the feature factory.

    If the file does not exist, this function remains without any effect.

    Parameters
    ----------
    nprocs : int
        The number of processors to set in the settings file.
    modify_ispecial : bool
        Whether to modify the 'ispecial' setting to 1.

    Returns
    -------
    None
    """
    _path = os.path.join(os.getcwd(), "settings.ini")
    if not os.path.isfile(_path):
        return

    with open(_path, "r") as f:
        lines = f.readlines()

    _start_threads = "  nthreads="
    _start_ispecial = "  ispecial="
    for line_idx, line in enumerate(lines):
        if line.startswith(_start_threads):
            splitted = line.split()
            lines[line_idx] = f"{_start_threads} {nprocs} {' '.join(splitted[2:])}\n"

        if line.startswith(_start_ispecial) and modify_ispecial is True:
            splitted = line.split()
            lines[line_idx] = f"{_start_ispecial} 1 {' '.join(splitted[2:])}\n"

    with open(_path, "w") as f:
        f.write("".join(lines))


def multiwfn_driver(
    cmds: List[Union[str, int, float]],
    input_file_path: str,
    output_file_name: str,
    environment_variables: Dict[str, Optional[str]],
    namespace: str,
    modify_ispecial: bool = False,
) -> None:
    """Run ``Multiwfn`` with the provided commands and environment variables.

    Parameters
    ----------
    cmds : List[Union[str, int, float]]
        A list of commands to be executed in Multiwfn.
    input_file_path : str
        The path to the input file for Multiwfn.
    output_file_name : str
        The name of the output file to save the results from Multiwfn.
    environment_variables : Dict[str, Optional[str]]
        A dictionary containing the environment variables to set before running Multiwfn with the
        respective values.
    namespace : str
        The namespace of the currently handled molecule for logging purposes.
    modify_ispecial : bool, optional
        Whether to modify the 'ispecial' setting in the Multiwfn settings file to 1. Default is
        ``False``.

    Returns
    -------
    None
    """
    _loc = get_function_or_method_name()

    # Check if Multiwfn program is available
    _call = "Multiwfn_noGUI"
    program_path = check_dependency_path(prg_name=_call)

    # Handle environment variables
    env = Environment(**environment_variables)
    env.set_environment()

    # Get Multiwfn settings file
    _path = os.path.join(os.path.dirname(program_path), "settings.ini")
    if os.path.isfile(_path):
        shutil.copy2(_path, os.getcwd())

        # Modify settings file and add the requested number of processors
        _modify_settings_ini(nprocs=getattr(env, "NUM_THREADS"), modify_ispecial=modify_ispecial)

    else:
        logging.warning(
            f"'{namespace}' | {_loc}()\nMultiwfn settings file (settings.ini) not found at "
            f"'{_path}'. Ensure that Multiwfn is installed correctly and run with the correct "
            "settings."
        )

    # Build Multiwfn command and run it
    command_string = "\n".join(map(str, cmds)) + "\n"
    res = run(
        [_call, input_file_path],
        input=command_string.encode(),
        stdout=PIPE,
        stderr=PIPE,
        check=False,
    )

    # Write output file
    command_string_ = command_string.replace("\n", " \\n ")[:-1]
    with open(f"{output_file_name}.out", "wb") as f:
        header = f"# Multiwfn commands: {command_string_}\n\n".encode("utf-8")
        f.write(header + res.stdout)

    # Reset environment
    env.reset_environment()

    # Clean up
    clean_up(to_be_removed=["settings.ini"])


def kallisto_driver(
    input_section: List[str], input_file_path: str, output_file_name: str
) -> Tuple[str, str]:
    """Run ``kallisto`` with the provided input section.

    Parameters
    ----------
    input_section : List[str]
        The input for kallisto to request the respective functionality.
    input_file_path : str
        The path to the input file for kallisto.
    output_file_name : str
        The name of the output file to save the results from kallisto.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the standard output and standard error from the kallisto call.
    """
    # Check if kallisto is available
    _call = "kallisto"
    _ = check_dependency_path(prg_name=_call)

    # Run kallisto
    kallisto_command = [_call]
    kallisto_command.extend(input_section)
    kallisto_command.extend(["--out", f"{output_file_name}.out", input_file_path])

    res = run(
        kallisto_command,
        stdout=PIPE,
        stderr=PIPE,
        check=False,
    )
    stdout = res.stdout.decode()
    stderr = res.stderr.decode()

    # Post-modify output file with kallisto call
    if os.path.isfile(f"{output_file_name}.out"):
        with open(f"{output_file_name}.out", "r") as f:
            lines = f.readlines()[::-1]

        lines.append(f"# kallisto program call: {' '.join(kallisto_command)}\n\n")
        lines = lines[::-1]

        with open(f"{output_file_name}.out", "w") as f:
            f.writelines(lines)

    return stdout, stderr


def external_driver(
    program_path: str,
    program_input: str,
    input_file_extension: str,
    namespace: str,
    dependencies: List[str] = [],
    **run_kwargs: Any,
) -> CompletedProcess[Any]:
    """Run an external program with the provided input as subprocess.

    This could either be a Python script (with ``.py`` extension) which is executed in a separate
    Python environment or any other external program (e.g., a compiled binary).

    Parameters
    ----------
    program_path : str
        The path to the external Python interpreter or program.
    program_input : str
        The input to the external program as a string.
    input_file_extension : str
        The file extension to use for the temporarily created input file (with the leading dot).
    namespace : str
        The namespace of the currently handled molecule for logging purposes.
    dependencies : List[str], optional
        A list of package names that are required in the external environment.
    **run_kwargs
        Optional additional keyword arguments to pass to ``subprocess.run``.

    Returns
    -------
    CompletedProcess
        The ``CompletedProcess`` instance from the ``subprocess.run`` call.
    """
    _loc = get_function_or_method_name()

    # Check input type
    _inpt = type(program_path)
    if _inpt != str:
        _errmsg = (
            f"Invalid input to 'program_path': must be of type str but obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {_loc}()\n{_errmsg}")
        raise TypeError(f"{_loc}(): {_errmsg}")

    _inpt = type(program_input)
    if _inpt != str:
        _errmsg = (
            f"Invalid input to 'program_input': must be of type str but obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {_loc}()\n{_errmsg}")
        raise TypeError(f"{_loc}(): {_errmsg}")

    _inpt = type(input_file_extension)
    if _inpt != str:
        _errmsg = (
            f"Invalid input to 'input_file_extension': must be of type str "
            f"but obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {_loc}()\n{_errmsg}")
        raise TypeError(f"{_loc}(): {_errmsg}")

    _inpt = type(namespace)
    if _inpt != str:
        _errmsg = (
            f"Invalid input to 'namespace': must be of type str but obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {_loc}()\n{_errmsg}")
        raise TypeError(f"{_loc}(): {_errmsg}")

    if isinstance(dependencies, list) is False:
        _errmsg = (
            f"Invalid input to 'dependencies': must be of type list but obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {_loc}()\n{_errmsg}")
        raise TypeError(f"{_loc}(): {_errmsg}")

    for entry in dependencies:
        _inpt = type(entry)
        if _inpt != str:
            _errmsg = (
                f"Invalid input to 'dependencies': all list entries must be of type str "
                f"but obtained {_inpt.__name__}."
            )
            logging.error(f"'{namespace}' | {_loc}()\n{_errmsg}")
            raise TypeError(f"{_loc}(): {_errmsg}")

    # Check dependency
    if input_file_extension == ".py":
        program_path = check_dependency_env(
            python_path=program_path, package_names=dependencies, namespace=namespace
        )
    else:
        program_path = check_dependency_path(prg_name=program_path)

    # Write the script to a temporary file
    with tempfile.NamedTemporaryFile("w", suffix=input_file_extension, delete=False) as f:
        f.write(program_input)
        tmp = f.name

    # Run the script
    try:
        res = run([program_path, tmp], **run_kwargs)
    except Exception as e:
        _errmsg = (
            f"An unexpected error occurred while running the external program "
            f"at '{program_path}': {e}"
        )
        logging.error(f"'{namespace}' | {_loc}()\n{_errmsg}")
        raise RuntimeError(f"{_loc}(): {_errmsg}")

    return res
