"""Molecule-level properties."""

import os
from typing import Dict, List, Optional, Tuple

from bonafide.utils.constants import KJ_MOL_TO_EV
from bonafide.utils.driver import multiwfn_driver


def _read_fmo_energies(
    multiplicity: int, file_lines: List[str]
) -> Tuple[Optional[float], Optional[float]]:
    """Read the HOMO and LUMO energy from a Multiwfn output file.

    Parameters
    ----------
    multiplicity : int
        The multiplicity of the molecule; required to correctly parse the Multiwfn output file.
    file_lines : List[str]
        The lines of the Multiwfn output file.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        The HOMO and LUMO energy as a tuple, or (``None``, ``None``) if not found.
    """
    homo_energy = None
    lumo_energy = None

    # Closed-shell case
    if multiplicity == 1:
        for line in file_lines:
            if "Note: Orbital" in line and "is HOMO, energy:" in line:
                try:
                    homo_energy = float(line.split()[-2])
                except:
                    pass
            if "Orbital" in line and "is LUMO, energy:" in line:
                try:
                    lumo_energy = float(line.split()[-2])
                except:
                    pass
        return homo_energy, lumo_energy

    # Open-shell case
    homo_a = None
    homo_b = None
    lumo_a = None
    lumo_b = None

    for line_idx, line in enumerate(file_lines):
        if "Note: Orbital" in line and "is alpha-HOMO, energy:" in line:
            try:
                homo_a = float(line.split()[-2])
            except:
                pass
            try:
                homo_b = float(file_lines[line_idx + 1].split()[-2])
            except:
                pass
            try:
                lumo_a = float(file_lines[line_idx + 2].split()[-2])
            except:
                pass
            try:
                lumo_b = float(file_lines[line_idx + 3].split()[-2])
            except:
                pass
            break

    if any([homo_a is None, homo_b is None, lumo_a is None, lumo_b is None]):
        return homo_energy, lumo_energy

    assert homo_a is not None  # for type checker
    assert homo_b is not None  # for type checker
    assert lumo_a is not None  # for type checker
    assert lumo_b is not None  # for type checker

    homo_energy = max([homo_a, homo_b])
    lumo_energy = min([lumo_a, lumo_b])
    return homo_energy, lumo_energy


def get_fmo_energies_multiwfn(
    input_file_path: str,
    output_file_name: str,
    multiplicity: int,
    environment_variables: Dict[str, Optional[str]],
    namespace: str,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Calculate the energy of the highest occupied and the lowest unoccupied molecular orbital
    energy from a Multiwfn output file.

    Parameters
    ----------
    input_file_path : str
        The path to the input file for running Multiwfn.
    output_file_name : str
        The name of the output file to which Multiwfn will write its results (without file
        extension).
    multiplicity : int
        The multiplicity of the molecule; required to correctly parse the Multiwfn output file.
    environment_variables : Dict[str, Optional[str]]
        A dictionary containing the environment variables to set before running Multiwfn with the
        respective values.
    namespace : str
        The namespace of the currently handled molecule for logging purposes.

    Returns
    -------
    Tuple[Optional[float], Optional[float], Optional[str]]
        HOMO and LUMO energy as well as an error message, which is ``None`` if everything worked
        as expected.
    """
    # Initialize target quantities as None
    homo_energy = None
    lumo_energy = None
    _errmsg = None

    # Use Multiwfn to read the HOMO and LUMO energy from the electronic structure file
    multiwfn_driver(
        cmds=[0, "q"],
        input_file_path=input_file_path,
        output_file_name=output_file_name,
        environment_variables=environment_variables,
        namespace=namespace,
    )

    # Check if the output file exists
    _opath = f"{output_file_name}.out"
    if os.path.isfile(_opath) is False:
        _errmsg = (
            f"Multiwfn output file '{_opath}' not found; probably the calculation "
            "did not run. Check your input"
        )
        return homo_energy, lumo_energy, _errmsg

    # Open output file
    with open(_opath, "r") as f:
        lines = f.readlines()

    # Read data from file
    homo_energy, lumo_energy = _read_fmo_energies(multiplicity=multiplicity, file_lines=lines)

    if homo_energy is None:
        _errmsg = f"HOMO energy could not be read from Multiwfn output file '{_opath}'"
    if lumo_energy is None:
        _errmsg = f"LUMO energy could not be read from Multiwfn output file '{_opath}'"

    return homo_energy, lumo_energy, _errmsg


def calculate_global_cdft_descriptors_fmo(
    homo_energy: float, lumo_energy: float
) -> Tuple[
    Optional[str],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """Calculate various conceptual DFT molecular descriptors from the HOMO and LUMO energy.

    Parameters
    ----------
    homo_energy : float
        The energy of the highest occupied molecular orbital (HOMO).
    lumo_energy : float
        The energy of the lowest unoccupied molecular orbital (LUMO).

    Returns
    -------
    Tuple[Optional[str], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]
        A tuple containing

        * an error message (``None`` if everything worked as expected),
        * HOMO-LUMO gap,
        * chemical potential,
        * hardness,
        * softness,
        * electrophilicity, and
        * nucleophilicity.

        The values are ``None`` if the calculation failed.
    """
    _errmsg = None

    homo_lumo_gap = None
    chem_potential = None
    hardness = None
    softness = None
    electrophilicity = None
    nucleophilicity = None

    try:
        homo_lumo_gap = float(lumo_energy - homo_energy)
        chem_potential = float((homo_energy + lumo_energy) / 2)
        hardness = float(homo_lumo_gap / 2)
        softness = float(1 / hardness)
        electrophilicity = float(chem_potential**2 / (2 * hardness))
        nucleophilicity = float(1 / electrophilicity)
    except Exception as e:
        _errmsg = (
            "calculation of global C-DFT descriptors from frontier molecular orbital "
            f"energies failed: {e}"
        )

    return (
        _errmsg,
        homo_lumo_gap,
        chem_potential,
        hardness,
        softness,
        electrophilicity,
        nucleophilicity,
    )


def calculate_global_cdft_descriptors_redox(
    energy_n: Tuple[float, str],
    energy_n_minus1: Tuple[float, str],
    energy_n_plus1: Tuple[float, str],
) -> Tuple[
    Optional[str],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """Calculate various conceptual DFT molecular descriptors from the ionization potential and
    electron affinity.

    All provided energies are expected to be in kJ/mol and are converted to eV.

    Parameters
    ----------
    energy_n : Tuple[float, str]
        The energy of the actual molecule that was calculated or provided by the user as value
        unit pair.
    energy_n_minus1 : Tuple[float, str]
        The energy of the one-electron-oxidized molecule that was calculated or provided by the
        user as value unit pair.
    energy_n_plus1 : Tuple[float, str]
        The energy of the one-electron-reduced molecule that was calculated or provided by the
        user as value unit pair.

    Returns
    -------
    Tuple[Optional[str], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]
        A tuple containing

        * an error message (``None`` if everything worked as expected),
        * ionization potential,
        * electron affinity,
        * chemical potential,
        * hardness,
        * softness,
        * electrophilicity, and
        * nucleophilicity.

        The values are ``None`` if the calculation failed.
    """
    _errmsg = None

    ionization_potential = None
    electron_affinity = None
    chem_potential = None
    hardness = None
    softness = None
    electrophilicity = None
    nucleophilicity = None

    try:
        # Convert energies from kJ/mol to eV
        energy_n_value = energy_n[0] * KJ_MOL_TO_EV
        energy_n_minus1_value = energy_n_minus1[0] * KJ_MOL_TO_EV
        energy_n_plus1_value = energy_n_plus1[0] * KJ_MOL_TO_EV

        ionization_potential = float(energy_n_minus1_value - energy_n_value)
        electron_affinity = float(-(energy_n_plus1_value - energy_n_value))
        chem_potential = float(-(ionization_potential + electron_affinity) / 2)
        hardness = float((ionization_potential - electron_affinity) / 2)
        softness = float(1 / hardness)
        electrophilicity = float(chem_potential**2 / (2 * hardness))
        nucleophilicity = float(-ionization_potential)
    except Exception as e:
        _errmsg = f"calculation of global C-DFT descriptors from redox energies failed: {e}"

    return (
        _errmsg,
        ionization_potential,
        electron_affinity,
        chem_potential,
        hardness,
        softness,
        electrophilicity,
        nucleophilicity,
    )
