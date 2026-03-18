"""Test functions for the ``bonafide.utils.sp_psi4`` module."""

import os
import shutil
from typing import Dict, List

import numpy as np
import pytest

from bonafide.utils.sp_psi4 import Psi4SP

# Data is from fluoromethanol.xyz
ELEMENTS = [
    "O",
    "C",
    "F",
    "H",
    "H",
    "H",
]

COORDINATES = np.array(
    [
        [0.84258, -0.69380, -0.04401],
        [-0.32457, 0.09824, -0.00222],
        [-1.37720, -0.74138, -0.14083],
        [1.59065, -0.08484, 0.05589],
        [-0.40257, 0.60940, 0.96014],
        [-0.32889, 0.81237, -0.82897],
    ]
)


PSI4_INPUT_PARAMS = {
    "method": "b3lyp-d3(bj)",
    "basis": "6-31g(d)",
    "solvent": "none",
    "solvent_model_solver": "iefpcm",
}

PSI4_INPUT_PARAMS2 = {
    "method": "pbe0",
    "basis": "def2-msvp",
    "solvent": "n-heptane",
    "solvent_model_solver": "iefpcm",
}

PSI4_INPUT_PARAMS3 = {
    "method": "m06-2x-d3zeroatm",
    "basis": "sto-3g",
    "solvent": " tetrahydrofurane",
    "solvent_model_solver": "cpcm",
}

####################################
# Test for the calculate() method. #
####################################


@pytest.mark.sp_psi4_calculate
@pytest.mark.parametrize(
    "write_el_struc_file, psi4_input_params, _expected_energy, _expected_strings_in_lines",
    [
        (
            True,
            PSI4_INPUT_PARAMS,
            -564351.1305966321,
            ["Name: 6-31G(D)", "=> B3LYP-D3(BJ): Empirical Dispersion <="],
        ),
        (
            False,
            PSI4_INPUT_PARAMS,
            -564351.1305966321,
            ["Name: 6-31G(D)", "=> B3LYP-D3(BJ): Empirical Dispersion <="],
        ),
        (
            True,
            PSI4_INPUT_PARAMS2,
            -563730.994145226,
            [
                "Name: DEF2-MSVP",
                "=> Composite Functional: PBE0 <=",
                "Solver Type: IEFPCM, isotropic",
                "Solvent name:          N-heptane",
            ],
        ),
        (
            True,
            PSI4_INPUT_PARAMS3,
            -556510.7685941831,
            [
                "Name: STO-3G",
                "=> M06-2X-D3ATM: Empirical Dispersion <=",
                "Solver Type: C-PCM",
                "Solvent name:          Tetrahydrofurane",
            ],
        ),
    ],
)
def test_calculate(
    write_el_struc_file: bool,
    psi4_input_params: Dict[str, str],
    _expected_energy: float,
    _expected_strings_in_lines: List[str],
) -> None:
    """Test for the ``calculate()`` method."""
    for charge, multiplicity in [(0, 1), (1, 2)]:
        # Initialize Psi4 single-point energy calculation class
        sp = Psi4SP()
        assert sp.engine_name == "Psi4"

        # Mock attributes required for calculation
        sp.conformer_name = "irrelevant_conformer_name"
        sp.elements = ELEMENTS
        sp.coordinates = COORDINATES
        sp.charge = charge
        sp.multiplicity = multiplicity
        sp.state = "n"
        sp.memory = "2 gb"
        sp.num_threads = 4
        sp.maxiter = 250

        for param, value in psi4_input_params.items():
            setattr(sp, param, value)

        # Make temporary output directory and change to it
        _dir_name = "_temp_psi4_test_dir"
        _out_file_name = f"Psi4SP__{sp.conformer_name}__{sp.state}"
        os.mkdir(_dir_name)
        os.chdir(_dir_name)

        # Run the single-point energy calculation
        energy, molden_file_path = sp.calculate(write_el_struc_file=write_el_struc_file)

        # Only check energy for closed-shell case
        if charge == 0:
            assert energy == pytest.approx(_expected_energy)

        if write_el_struc_file is True:
            assert os.path.isfile(f"{_out_file_name}.molden") is True
            with open(f"{_out_file_name}.molden", "r") as f:
                assert len(f.read()) > 0
            assert molden_file_path == os.path.join(
                os.path.split(os.getcwd())[0], f"{_out_file_name}.molden"
            )
        else:
            assert os.path.isfile(f"{_out_file_name}.molden") is False
            assert molden_file_path is None

        # Check output file
        assert os.path.isfile(f"{_out_file_name}.out") is True
        with open(f"{_out_file_name}.out", "r") as f:
            lines = f.readlines()

        _expected_strings_in_lines.extend(
            ["Energy and wave function converged.", "Computation Completed"]
        )
        for expected_str in _expected_strings_in_lines:
            assert any([expected_str in line for line in lines])

        # Clean up
        os.chdir("..")
        shutil.rmtree(_dir_name)
