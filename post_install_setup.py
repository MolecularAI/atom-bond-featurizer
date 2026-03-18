"""Modify ``qmdesc`` to make it compatible with PyTorch 2.0+."""

import os
import warnings

warnings.filterwarnings(action="ignore")


def _get_path() -> str:
    """Get the installation path of qmdesc.

    Returns
    -------
    str
        The path to the qmdesc installation directory.
    """
    try:
        import qmdesc
    except ImportError:
        raise ImportError("qmdesc must be installed to run this script.")

    return str(os.path.dirname(qmdesc.__file__))


def modify_script(path: str) -> None:
    """Modify the ``handler.py`` script in qmdesc.

    This is needed to ensure compatibility with the latest versions of PyTorch.

    Parameters
    ----------
    path : str
        The path to the qmdesc installation directory.
    """
    path = os.path.join(path, "handler.py")
    with open(path, "r") as file:
        lines = file.readlines()

    if len(lines) < 34:
        raise IndexError("The handler.py script does not have the expected number of lines.")

    _old = "        state = torch.load(stream, lambda storage, loc: storage)\n"
    _new = "        state = torch.load(stream, lambda storage, loc: storage, weights_only=False)\n"

    if lines[33] == _new:
        return

    if lines[33] != _old:
        raise ValueError("The handler.py script does not have the expected format.")

    lines[33] = _new

    with open(path, "w") as file:
        file.writelines(lines)


if __name__ == "__main__":
    qmdesc_path = _get_path()
    modify_script(qmdesc_path)
    print("\n ==> Post-installation setup successful. \n")
