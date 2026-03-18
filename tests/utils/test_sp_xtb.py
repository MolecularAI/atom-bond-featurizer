"""Test functions for the ``bonafide.utils.sp_xtb`` module."""

import os
import shutil
from typing import Dict, List, Optional

import numpy as np
import pytest

from bonafide.utils.sp_xtb import XtbSP

# Data is from clopidogrel-conf_05.xyz
ELEMENTS = [
    "C",
    "O",
    "C",
    "C",
    "N",
    "C",
    "C",
    "C",
    "C",
    "C",
    "C",
    "C",
    "S",
    "C",
    "C",
    "C",
    "C",
    "C",
    "C",
    "Cl",
    "O",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
]

COORDINATES = np.array(
    [
        [1.32563, 3.15106, 2.07812],
        [0.75235, 1.90289, 1.69056],
        [1.01099, 1.54790, 0.39414],
        [0.37426, 0.16361, 0.09068],
        [-0.04735, 0.04828, -1.34904],
        [-0.67235, -1.26552, -1.66503],
        [-2.01217, -1.56212, -0.95108],
        [-2.83220, -0.31425, -0.89333],
        [-2.35219, 0.93464, -1.23646],
        [-0.96126, 1.14151, -1.76826],
        [-3.35004, 1.93734, -1.03788],
        [-4.52871, 1.40922, -0.55417],
        [-4.43594, -0.28740, -0.34014],
        [1.39737, -0.90370, 0.48712],
        [2.54100, -1.09356, -0.31833],
        [3.51302, -2.03752, 0.01521],
        [3.36815, -2.80581, 1.16514],
        [2.25342, -2.62530, 1.98268],
        [1.27593, -1.67886, 1.65228],
        [-0.05739, -1.51911, 2.73923],
        [1.67008, 2.22647, -0.38660],
        [0.91895, 3.96502, 1.46989],
        [1.06171, 3.33509, 3.12319],
        [2.41653, 3.11332, 1.99821],
        [-0.52542, 0.09682, 0.70897],
        [0.02541, -2.09016, -1.48064],
        [-0.84477, -1.29307, -2.75054],
        [-1.81950, -1.91798, 0.06724],
        [-2.54454, -2.35487, -1.48857],
        [-0.58883, 2.13127, -1.48840],
        [-1.01047, 1.14668, -2.86518],
        [-3.21438, 2.99276, -1.24170],
        [-5.44245, 1.93585, -0.31559],
        [2.67434, -0.49610, -1.22095],
        [4.38333, -2.16791, -0.62359],
        [4.12428, -3.54093, 1.42875],
        [2.15322, -3.22554, 2.88405],
    ]
)

XTB_INPUT_PARAMS = {
    "method": "gfn2-xtb",
    "iterations": "500",
    "acc": "1",
    "etemp": "300",
    "solvent_model": "none",
    "solvent": "none",
}

XTB_INPUT_PARAMS2 = {
    "method": "gfn1-xtb",
    "iterations": "500",
    "acc": "1",
    "etemp": "300",
    "solvent_model": "none",
    "solvent": "none",
}

XTB_INPUT_PARAMS3 = {
    "method": "gfn0-xtb",
    "iterations": "200",
    "acc": "1",
    "etemp": "300",
    "solvent_model": "none",
    "solvent": "none",
}

XTB_INPUT_PARAMS4 = {
    "method": "gfn2-xtb",
    "iterations": "500",
    "acc": "1",
    "etemp": "300",
    "solvent_model": "alpb",
    "solvent": "methanol",
}

XTB_INPUT_PARAMS5 = {
    "method": "gfn1-xtb",
    "iterations": "500",
    "acc": "1",
    "etemp": "200",
    "solvent_model": "cosmo",
    "solvent": "octanol",
}

# Check if xtb is installed
if shutil.which("xtb") is None:
    raise EnvironmentError(
        "xtb executable not found. Please install xtb to run the following tests."
    )

# Set XTBHOME environment variable required for GFN0-xTB calculations
os.environ["XTBHOME"] = os.path.join(
    os.path.dirname(os.path.dirname(shutil.which("xtb"))), "share", "xtb"
)


####################################
# Test for the calculate() method. #
####################################


@pytest.mark.sp_xtb_calculate
@pytest.mark.parametrize(
    "write_el_struc_file, calc_fukui, calc_ceh, out_file_name, state, xtb_input_params, "
    "_expected_energy, _expected_strings_in_lines",
    [
        (
            False,
            False,
            False,
            "test_output1",
            None,
            XTB_INPUT_PARAMS,
            -160872.97776881297,
            ["Hamiltonian                  GFN2-xTB"],
        ),
        (
            False,
            False,
            False,
            "test_output2",
            None,
            XTB_INPUT_PARAMS2,
            -165378.93345060406,
            ["Hamiltonian                  GFN1-xTB"],
        ),
        (
            False,
            False,
            False,
            "test_output3",
            None,
            XTB_INPUT_PARAMS3,
            -156541.84664190828,
            [" G F N 0 - x T B "],
        ),
        (
            True,
            False,
            False,
            "test_output1",
            None,
            XTB_INPUT_PARAMS,
            -160872.97776881297,
            ["Hamiltonian                  GFN2-xTB"],
        ),
        (
            True,
            False,
            False,
            "test_output2",
            None,
            XTB_INPUT_PARAMS2,
            -165378.93345060406,
            ["Hamiltonian                  GFN1-xTB"],
        ),
        (
            True,
            False,
            False,
            "test_output3",
            None,
            XTB_INPUT_PARAMS3,
            -156541.84664190828,
            [" G F N 0 - x T B "],
        ),
        (
            True,
            True,
            False,
            "test_output7",
            None,
            XTB_INPUT_PARAMS,
            -160872.97776881297,
            ["Hamiltonian                  GFN2-xTB", "Fukui functions:"],
        ),
        (
            False,
            True,
            False,
            "test_output8",
            None,
            XTB_INPUT_PARAMS2,
            -165378.93345060406,
            ["Hamiltonian                  GFN1-xTB", "Fukui functions:"],
        ),
        (
            False,
            True,
            False,
            "test_output9",
            None,
            XTB_INPUT_PARAMS3,
            -156541.84664190828,
            [" G F N 0 - x T B ", "Fukui functions:"],
        ),
        (
            True,
            True,
            True,
            "test_output10",
            None,
            XTB_INPUT_PARAMS,
            -160872.97776881297,
            ["Hamiltonian                  GFN2-xTB", "Fukui functions:"],
        ),
        (
            True,
            True,
            True,
            None,
            "n",
            XTB_INPUT_PARAMS,
            -160872.97776881297,
            ["Hamiltonian                  GFN2-xTB", "Fukui functions:"],
        ),
        (
            True,
            True,
            True,
            "test_output12",
            None,
            XTB_INPUT_PARAMS4,
            -160938.09009006506,
            [
                "Hamiltonian                  GFN2-xTB",
                "Fukui functions:",
                "Solvent                        methanol",
                "Solvation model:               ALPB",
            ],
        ),
        (
            True,
            True,
            True,
            "test_output13",
            None,
            XTB_INPUT_PARAMS5,
            -165460.34730947582,
            [
                "Hamiltonian                  GFN1-xTB",
                "Fukui functions:",
                "Solvent                        octanol",
                "Solvation model:               COSMO",
            ],
        ),
    ],
)
def test_calculate(
    write_el_struc_file: bool,
    calc_fukui: bool,
    calc_ceh: bool,
    out_file_name: Optional[str],
    state: str,
    xtb_input_params: Dict[str, str],
    _expected_energy: float,
    _expected_strings_in_lines: List[str],
) -> None:
    """Test for the ``calculate()`` method."""
    for charge, multiplicity in [(0, 1), (1, 2)]:
        # Initialize xtb single-point energy calculation class
        sp = XtbSP()
        assert sp.engine_name == "xtb"

        # Mock attributes required for calculation
        sp.conformer_name = "irrelevant_conformer_name"
        sp.elements = ELEMENTS
        sp.coordinates = COORDINATES
        sp.charge = charge
        sp.multiplicity = multiplicity

        for param, value in xtb_input_params.items():
            setattr(sp, param, value)

        if out_file_name is None:
            sp.state = state
            _out_file_name = f"XtbSP__{sp.conformer_name}__{state}"
        else:
            _out_file_name = out_file_name

        # Make temporary output directory and change to it
        _dir_name = "_temp_xtb_test_dir"
        os.mkdir(_dir_name)
        os.chdir(_dir_name)

        # Run the single-point energy calculation
        energy, molden_file_path = sp.calculate(
            write_el_struc_file=write_el_struc_file,
            calc_fukui=calc_fukui,
            calc_ceh=calc_ceh,
            out_file_name=out_file_name,
        )

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

        if calc_ceh is True:
            assert os.path.isfile("ceh.charges") is True
            with open("ceh.charges", "r") as f:
                assert len(f.read()) > 0
        else:
            assert os.path.isfile("ceh.charges") is False

        # Check output file
        assert os.path.isfile(f"{_out_file_name}.out") is True
        with open(f"{_out_file_name}.out", "r") as f:
            lines = f.readlines()

        _expected_strings_in_lines.extend(["xtb version 6.7.1", "* finished run on"])
        for expected_str in _expected_strings_in_lines:
            assert any([expected_str in line for line in lines])

        # Clean up
        os.chdir("..")
        shutil.rmtree(_dir_name)
