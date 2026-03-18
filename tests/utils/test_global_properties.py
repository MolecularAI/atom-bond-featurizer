"""Test functions for the ``bonafide.utils.global_properties`` module."""

import os
from typing import Callable, Optional, Tuple, Union

import pytest

from bonafide.utils.global_properties import (
    calculate_global_cdft_descriptors_fmo,
    calculate_global_cdft_descriptors_redox,
    get_fmo_energies_multiwfn,
)

#######################################################
# Tests for the get_fmo_energies_multiwfn() function. #
#######################################################

MULTIWFN_ENV_VARS = {"OMP_STACKSIZE": "1G", "NUM_THREADS": 4}


@pytest.mark.get_fmo_energies_multiwfn
@pytest.mark.parametrize(
    "input_file_name, multiplicity, expected_fmo_values",
    [
        ("clopidogrel-conf_05.molden", 1, (-9.751016, -7.057492)),
        ("spiro_mol-conf_00.fchk", 1, (-6.276861, -1.576732)),
        ("fluoromethanol.fchk", 1, (-7.653921, 1.795297)),
        ("radical_cation-conf_00.fchk", 2, (-10.946884, -8.841420)),
        ("Cr_III_cation-conf_00.fchk", 4, (-9.975817, -5.063482)),
    ],
)
def test_get_fmo_energies_multiwfn(
    fetch_data_file: Callable[[str], str],
    input_file_name: str,
    multiplicity: int,
    expected_fmo_values: Tuple[float, float],
) -> None:
    """Test for the ``get_fmo_energies_multiwfn()`` function: valid input."""
    _input_path = fetch_data_file(input_file_name)
    _output_name = "irrelevant_name"

    # Calculate HOMO and LUMO energy
    homo, lumo, error_message = get_fmo_energies_multiwfn(
        input_file_path=_input_path,
        output_file_name=_output_name,
        multiplicity=multiplicity,
        environment_variables=MULTIWFN_ENV_VARS,
        namespace="irrelevant_namespace_string",
    )

    # Check values
    assert homo == expected_fmo_values[0]
    assert lumo == expected_fmo_values[1]
    assert error_message is None

    # Clean up
    os.remove(f"{_output_name}.out")


@pytest.mark.get_fmo_energies_multiwfn
@pytest.mark.parametrize(
    "input_file_name, multiplicity",
    [
        ("clopidogrel-conf_05.molden", 3),
        ("spiro_mol-conf_00.fchk", 2),
        ("radical_cation-conf_00.fchk", 1),
    ],
)
def test_get_fmo_energies_multiwfn2(
    fetch_data_file: Callable[[str], str], input_file_name: str, multiplicity: int
) -> None:
    """Test for the ``get_fmo_energies_multiwfn()`` function: wrong multiplicity."""
    _input_path = fetch_data_file(input_file_name)
    _output_name = "irrelevant_name"

    # Calculate HOMO and LUMO energy
    homo, lumo, error_message = get_fmo_energies_multiwfn(
        input_file_path=_input_path,
        output_file_name=_output_name,
        multiplicity=multiplicity,
        environment_variables=MULTIWFN_ENV_VARS,
        namespace="irrelevant_namespace_string",
    )

    # Check values
    assert homo is None
    assert lumo is None
    assert "energy could not be read from Multiwfn output file" in error_message

    # Clean up
    os.remove(f"{_output_name}.out")


###################################################################
# Tests for the calculate_global_cdft_descriptors_fmo() function. #
###################################################################


@pytest.mark.calculate_global_cdft_descriptors_fmo
@pytest.mark.parametrize(
    "homo_energy, lumo_energy, expected_output",
    [
        (-2.0, -1.0, (None, 1.0, -1.5, 0.5, 2.0, 2.25, 0.4444444444444444)),
        (1, 2, (None, 1.0, 1.5, 0.5, 2.0, 2.25, 0.4444444444444444)),
        (
            -5.738759,
            -0.663565,
            (
                None,
                5.075194,
                -3.201162,
                2.537597,
                0.39407360585624907,
                2.0191224513277723,
                0.49526466279566217,
            ),
        ),
        (
            -7.653921,
            1.795297,
            (
                None,
                9.449218,
                -2.9293120000000004,
                4.724609,
                0.2116577265970581,
                0.9081035905134163,
                1.1011959543455039,
            ),
        ),
        (
            -2.0,
            -2.0,
            (
                "calculation of global C-DFT descriptors from frontier molecular orbital "
                "energies failed:",
                0.0,
                -2.0,
                0.0,
                None,
                None,
                None,
            ),
        ),
        (
            "i_should_not_be_here",
            -2.0,
            (
                "calculation of global C-DFT descriptors from frontier molecular orbital "
                "energies failed:",
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ),
    ],
)
def test_calculate_global_cdft_descriptors(
    homo_energy: Union[int, float, str],
    lumo_energy: Union[int, float, str],
    expected_output: Tuple[
        Optional[str],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ],
) -> None:
    """Test for the ``calculate_global_cdft_descriptors_fmo()`` function."""
    expected_error_message = expected_output[0]
    expected_homo_lumo_gap = expected_output[1]
    expected_chem_potential = expected_output[2]
    expected_hardness = expected_output[3]
    expected_softness = expected_output[4]
    expected_electrophilicity = expected_output[5]
    expected_nucleophilicity = expected_output[6]

    (
        error_message,
        homo_lumo_gap,
        chem_potential,
        hardness,
        softness,
        electrophilicity,
        nucleophilicity,
    ) = calculate_global_cdft_descriptors_fmo(homo_energy=homo_energy, lumo_energy=lumo_energy)

    if expected_error_message is None:
        assert error_message is None
        assert homo_lumo_gap == pytest.approx(expected_homo_lumo_gap)
        assert chem_potential == pytest.approx(expected_chem_potential)
        assert hardness == pytest.approx(expected_hardness)
        assert softness == pytest.approx(expected_softness)
        assert electrophilicity == pytest.approx(expected_electrophilicity)
        assert nucleophilicity == pytest.approx(expected_nucleophilicity)
    else:
        assert expected_error_message in error_message

        if expected_homo_lumo_gap is not None:
            assert homo_lumo_gap == pytest.approx(expected_homo_lumo_gap)
            assert chem_potential == pytest.approx(expected_chem_potential)
            assert hardness == pytest.approx(expected_hardness)
        else:
            assert homo_lumo_gap is None
            assert chem_potential is None
            assert hardness is None

        assert softness is None
        assert electrophilicity is None
        assert nucleophilicity is None


#####################################################################
# Tests for the calculate_global_cdft_descriptors_redox() function. #
#####################################################################


@pytest.mark.calculate_global_cdft_descriptors_redox
@pytest.mark.parametrize(
    "energy_n, energy_n_minus1, energy_n_plus1, expected_output",
    [
        (
            (10, "kj_mol"),
            (5, "kj_mol"),
            (20, "kj_mol"),
            (
                None,
                -0.05182135,
                -0.1036427,
                0.077732025,
                0.025910675,
                38.594131569324226,
                0.1165980375,
                0.05182135,
            ),
        ),
        (
            (-2493544.572202751, "kj_mol"),
            (-2492765.749240541, "kj_mol"),
            (-2493467.608306304, "kj_mol"),
            (
                None,
                8.071931462545763,
                -0.7976746030290087,
                -3.637128429758377,
                4.434803032787386,
                0.2254891576033479,
                1.4914645720175381,
                -8.071931462545763,
            ),
        ),
        (
            (-36254.125, "kj_mol"),
            (-36254.125, "kj_mol"),
            (-36254.125, "kj_mol"),
            (
                "calculation of global C-DFT descriptors from redox energies failed:",
                0.0,
                0.0,
                0.0,
                0.0,
                None,
                None,
                None,
            ),
        ),
        (
            ("i_should_not_be_here", "kj_mol"),
            (120.0, "kj_mol"),
            (80.0, "kj_mol"),
            (
                "calculation of global C-DFT descriptors from redox energies failed:",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ),
    ],
)
def test_calculate_global_cdft_descriptors_redox(
    energy_n: Tuple[Union[int, float, str], str],
    energy_n_minus1: Tuple[Union[int, float, str], str],
    energy_n_plus1: Tuple[Union[int, float, str], str],
    expected_output: Tuple[
        Optional[str],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ],
) -> None:
    """Test for the ``calculate_global_cdft_descriptors_redox()`` function."""
    expected_error_message = expected_output[0]
    expected_ionization_potential = expected_output[1]
    expected_electron_affinity = expected_output[2]
    expected_chem_potential = expected_output[3]
    expected_hardness = expected_output[4]
    expected_softness = expected_output[5]
    expected_electrophilicity = expected_output[6]
    expected_nucleophilicity = expected_output[7]

    (
        error_message,
        ionization_potential,
        electron_affinity,
        chem_potential,
        hardness,
        softness,
        electrophilicity,
        nucleophilicity,
    ) = calculate_global_cdft_descriptors_redox(
        energy_n=energy_n, energy_n_minus1=energy_n_minus1, energy_n_plus1=energy_n_plus1
    )

    if expected_error_message is None:
        assert error_message is None
        assert ionization_potential == pytest.approx(expected_ionization_potential)
        assert electron_affinity == pytest.approx(expected_electron_affinity)
        assert chem_potential == pytest.approx(expected_chem_potential)
        assert hardness == pytest.approx(expected_hardness)
        assert softness == pytest.approx(expected_softness)
        assert electrophilicity == pytest.approx(expected_electrophilicity)
        assert nucleophilicity == pytest.approx(expected_nucleophilicity)
    else:
        assert expected_error_message in error_message

        if expected_ionization_potential is not None:
            assert ionization_potential == pytest.approx(expected_ionization_potential)
            assert electron_affinity == pytest.approx(expected_electron_affinity)
            assert chem_potential == pytest.approx(expected_chem_potential)
            assert hardness == pytest.approx(expected_hardness)
        else:
            assert ionization_potential is None
            assert electron_affinity is None
            assert chem_potential is None
            assert hardness is None

        assert softness is None
        assert electrophilicity is None
        assert nucleophilicity is None
