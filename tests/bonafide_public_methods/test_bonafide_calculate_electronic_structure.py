"""Test functions for the ``calculate_electronic_structure()`` method in the ``bonafide.bonafide``
module.
"""

from __future__ import annotations

import logging
import os
import shutil
import warnings
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union
from unittest.mock import call

import pytest
from pytest_mock import MockerFixture
from rdkit import RDLogger

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer
    from bonafide.utils.molecule_vault import MolVault


# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


@pytest.mark.calculate_electronic_structure
@pytest.mark.parametrize(
    "input_file, n_conformers, engine_name, redox_state",
    [
        ("clopidogrel-conf_01.xyz", 1, "psi4", " N"),
        ("clopidogrel-conf_01.xyz", 1, "psi4", " N-1"),
        ("clopidogrel-conf_01.xyz", 1, "psi4", " N+1"),
        ("clopidogrel-conf_01.xyz", 1, "psi4", "all"),
        ("clopidogrel-conf_01.xyz", 1, "xtb", "n"),
        ("clopidogrel-conf_01.xyz", 1, "xtb", "n-1"),
        ("clopidogrel-conf_01.xyz", 1, "xtb", "n+1"),
        ("clopidogrel-conf_01.xyz", 1, "xtb", "all"),
        ("clopidogrel.xyz", 7, "Psi4", "n"),
        ("clopidogrel.xyz", 7, "Psi4", "n-1"),
        ("clopidogrel.xyz", 7, "Psi4", "n+1"),
        ("clopidogrel.xyz", 7, "Psi4", "all"),
        ("clopidogrel.xyz", 7, "XTB   ", "n"),
        ("clopidogrel.xyz", 7, "XTB   ", "n-1"),
        ("clopidogrel.xyz", 7, "XTB   ", "n+1"),
        ("clopidogrel.xyz", 7, "XTB   ", "all"),
    ],
)
def test_calculate_electronic_structure(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    input_file: str,
    n_conformers: int,
    engine_name: str,
    redox_state: str,
) -> None:
    """Test for the ``calculate_electronic_structure()`` method: valid inputs."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name=input_file)
    f.read_input(
        input_value=input_string,
        namespace="irrelevant",
        input_format="file",
        output_directory="irrelevant_out_dir",
    )
    f.set_charge(charge=0)
    f.set_multiplicity(multiplicity=1)
    assert f.mol_vault is not None
    assert f.mol_vault.is_valid == [True] * n_conformers
    assert f.mol_vault.energies_n == []
    assert f.mol_vault.energies_n_minus1 == []
    assert f.mol_vault.energies_n_plus1 == []
    assert f.mol_vault.energies_n_read is False
    assert f.mol_vault.energies_n_minus1_read is False
    assert f.mol_vault.energies_n_plus1_read is False
    assert f.mol_vault.electronic_strucs_n == []
    assert f.mol_vault.electronic_strucs_n_minus1 == []
    assert f.mol_vault.electronic_strucs_n_plus1 == []
    assert f.mol_vault.electronic_struc_types_n == []
    assert f.mol_vault.electronic_struc_types_n_minus1 == []
    assert f.mol_vault.electronic_struc_types_n_plus1 == []

    # Mock the actual single-point energy calculation and the generated electronic structure file
    _mock_el_struc_file = os.path.join(
        os.getcwd(), "irrelevant_out_dir", "dummy_el_struc_file.molden"
    )
    with open(_mock_el_struc_file, "w") as file:
        file.write("Nothing to see here.\n")

    _mocked_energies = [(-123.123, "kj_mol")] * n_conformers
    _mocked_electronic_strucs = [_mock_el_struc_file] * n_conformers
    _mocked_is_valid_list = [True] * n_conformers
    _expected_electronic_struc_types = ["molden"] * n_conformers

    if engine_name.strip().lower() == "psi4":
        _patch_path = "bonafide.utils.sp_psi4.Psi4SP.run"
    if engine_name.strip().lower() == "xtb":
        _patch_path = "bonafide.utils.sp_xtb.XtbSP.run"

    mock = mocker.patch(_patch_path, return_value=(_mocked_energies, _mocked_electronic_strucs))

    # Calculate electronic structure
    f.calculate_electronic_structure(engine=engine_name, redox=redox_state)

    # Check mol vault based on the requested redox state
    redox_state = redox_state.strip().lower()

    assert f.mol_vault.is_valid == _mocked_is_valid_list

    if redox_state == "n":
        mock.assert_called_with(state=redox_state, write_el_struc_file=True)
        assert mock.call_count == 1

        assert f.mol_vault.energies_n == _mocked_energies
        assert f.mol_vault.energies_n_minus1 == []
        assert f.mol_vault.energies_n_plus1 == []
        assert f.mol_vault.energies_n_read is True
        assert f.mol_vault.energies_n_minus1_read is False
        assert f.mol_vault.energies_n_plus1_read is False
        assert f.mol_vault.electronic_strucs_n == _mocked_electronic_strucs
        assert f.mol_vault.electronic_strucs_n_minus1 == []
        assert f.mol_vault.electronic_strucs_n_plus1 == []
        assert f.mol_vault.electronic_struc_types_n == _expected_electronic_struc_types
        assert f.mol_vault.electronic_struc_types_n_minus1 == []
        assert f.mol_vault.electronic_struc_types_n_plus1 == []

    elif redox_state == "n-1":
        mock.assert_has_calls(
            [call(state="n", write_el_struc_file=True), call(state="n-1", write_el_struc_file=True)]
        )
        assert mock.call_count == 2

        assert f.mol_vault.energies_n == _mocked_energies
        assert f.mol_vault.energies_n_minus1 == _mocked_energies
        assert f.mol_vault.energies_n_plus1 == []
        assert f.mol_vault.energies_n_read is True
        assert f.mol_vault.energies_n_minus1_read is True
        assert f.mol_vault.energies_n_plus1_read is False
        assert f.mol_vault.electronic_strucs_n == _mocked_electronic_strucs
        assert f.mol_vault.electronic_strucs_n_minus1 == _mocked_electronic_strucs
        assert f.mol_vault.electronic_strucs_n_plus1 == []
        assert f.mol_vault.electronic_struc_types_n == _expected_electronic_struc_types
        assert f.mol_vault.electronic_struc_types_n_minus1 == _expected_electronic_struc_types
        assert f.mol_vault.electronic_struc_types_n_plus1 == []

    elif redox_state == "n+1":
        mock.assert_has_calls(
            [call(state="n", write_el_struc_file=True), call(state="n+1", write_el_struc_file=True)]
        )
        assert mock.call_count == 2

        assert f.mol_vault.energies_n == _mocked_energies
        assert f.mol_vault.energies_n_minus1 == []
        assert f.mol_vault.energies_n_plus1 == _mocked_energies
        assert f.mol_vault.energies_n_read is True
        assert f.mol_vault.energies_n_minus1_read is False
        assert f.mol_vault.energies_n_plus1_read is True
        assert f.mol_vault.electronic_strucs_n == _mocked_electronic_strucs
        assert f.mol_vault.electronic_strucs_n_minus1 == []
        assert f.mol_vault.electronic_strucs_n_plus1 == _mocked_electronic_strucs
        assert f.mol_vault.electronic_struc_types_n == _expected_electronic_struc_types
        assert f.mol_vault.electronic_struc_types_n_minus1 == []
        assert f.mol_vault.electronic_struc_types_n_plus1 == _expected_electronic_struc_types

    elif redox_state == "all":
        mock.assert_has_calls(
            [
                call(state="n", write_el_struc_file=True),
                call(state="n-1", write_el_struc_file=True),
                call(state="n+1", write_el_struc_file=True),
            ]
        )
        assert mock.call_count == 3

        assert f.mol_vault.energies_n == _mocked_energies
        assert f.mol_vault.energies_n_minus1 == _mocked_energies
        assert f.mol_vault.energies_n_plus1 == _mocked_energies
        assert f.mol_vault.energies_n_read is True
        assert f.mol_vault.energies_n_minus1_read is True
        assert f.mol_vault.energies_n_plus1_read is True
        assert f.mol_vault.electronic_strucs_n == _mocked_electronic_strucs
        assert f.mol_vault.electronic_strucs_n_minus1 == _mocked_electronic_strucs
        assert f.mol_vault.electronic_strucs_n_plus1 == _mocked_electronic_strucs
        assert f.mol_vault.electronic_struc_types_n == _expected_electronic_struc_types
        assert f.mol_vault.electronic_struc_types_n_minus1 == _expected_electronic_struc_types
        assert f.mol_vault.electronic_struc_types_n_plus1 == _expected_electronic_struc_types

    else:
        raise ValueError(f"Unexpected redox state '{redox_state}' encountered in test.")

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)

    # Clean up
    shutil.rmtree(path="irrelevant_out_dir")
    clean_up_logfile()


@pytest.mark.calculate_electronic_structure
def test_calculate_electronic_structure2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``calculate_electronic_structure()`` method: fails because no molecule
    read yet.
    """
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Calculate electronic structure
    with pytest.raises(ValueError, match="calculating electronic structures"):
        f.calculate_electronic_structure(engine="xtb")
    assert f.mol_vault is None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.calculate_electronic_structure
def test_calculate_electronic_structure3(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``calculate_electronic_structure()`` method: fails because molecule in vault
    is 2D (which means that the electronic structure cannot be calculated).
    """
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    f.read_input(input_value="O(C([H])([H])F)[H]", namespace="irrelevant")
    assert f.mol_vault is not None

    # Calculate electronic structure
    with pytest.raises(
        ValueError, match="Electronic structure calculations are not feasible for 2D ensembles."
    ):
        f.calculate_electronic_structure(engine="psi4")
    assert f.mol_vault is not None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.calculate_electronic_structure
@pytest.mark.parametrize(
    "engine, redox, _set_charge, _set_multiplicity, _engine_name, _error_type",
    [
        (["xtb", "psi4"], "n", True, True, "psi4", TypeError),
        ("xtb", False, True, True, "xtb", TypeError),
        ("xtb", "n", False, True, "xtb", ValueError),
        ("psi4", "n", True, False, "psi4", ValueError),
        ("the-real-xtb", "n+1", True, True, "xtb", ValueError),
        ("xtb", "n_1", True, True, "xtb", ValueError),
    ],
)
def test_calculate_electronic_structure4(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    engine: Any,
    redox: Any,
    _set_charge: bool,
    _set_multiplicity: bool,
    _engine_name: str,
    _error_type: Any,
) -> None:
    """Test for the ``calculate_electronic_structure()`` method: invalid inputs."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name="clopidogrel-conf_01.xyz")
    f.read_input(
        input_value=input_string,
        namespace="irrelevant",
        input_format="file",
        output_directory="irrelevant_out_dir",
    )
    assert f.mol_vault is not None
    assert f.mol_vault.is_valid == [True]
    assert f.mol_vault.energies_n == []
    assert f.mol_vault.energies_n_minus1 == []
    assert f.mol_vault.energies_n_plus1 == []
    assert f.mol_vault.energies_n_read is False
    assert f.mol_vault.energies_n_minus1_read is False
    assert f.mol_vault.energies_n_plus1_read is False
    assert f.mol_vault.electronic_strucs_n == []
    assert f.mol_vault.electronic_strucs_n_minus1 == []
    assert f.mol_vault.electronic_strucs_n_plus1 == []
    assert f.mol_vault.electronic_struc_types_n == []
    assert f.mol_vault.electronic_struc_types_n_minus1 == []
    assert f.mol_vault.electronic_struc_types_n_plus1 == []

    # Set charge and multiplicity if requested
    if _set_charge is True:
        f.set_charge(charge=0)
    if _set_multiplicity is True:
        f.set_multiplicity(multiplicity=1)

    # Mock the actual single-point energy calculation and the generated electronic structure file
    _mock_el_struc_file = os.path.join(
        os.getcwd(), "irrelevant_out_dir", "dummy_el_struc_file.molden"
    )
    with open(_mock_el_struc_file, "w") as file:
        file.write("Nothing to see here.\n")

    if _engine_name == "psi4":
        _patch_path = "bonafide.utils.sp_psi4.Psi4SP.run"
    if _engine_name == "xtb":
        _patch_path = "bonafide.utils.sp_xtb.XtbSP.run"

    mock = mocker.patch(_patch_path)

    # Calculate electronic structure
    with pytest.raises(_error_type):
        f.calculate_electronic_structure(engine=engine, redox=redox)
    mock.assert_not_called()

    assert f.mol_vault.is_valid == [True]
    assert f.mol_vault.energies_n == []
    assert f.mol_vault.energies_n_minus1 == []
    assert f.mol_vault.energies_n_plus1 == []
    assert f.mol_vault.energies_n_read is False
    assert f.mol_vault.energies_n_minus1_read is False
    assert f.mol_vault.energies_n_plus1_read is False
    assert f.mol_vault.electronic_strucs_n == []
    assert f.mol_vault.electronic_strucs_n_minus1 == []
    assert f.mol_vault.electronic_strucs_n_plus1 == []
    assert f.mol_vault.electronic_struc_types_n == []
    assert f.mol_vault.electronic_struc_types_n_minus1 == []
    assert f.mol_vault.electronic_struc_types_n_plus1 == []

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    shutil.rmtree(path="irrelevant_out_dir")
    clean_up_logfile()


@pytest.mark.calculate_electronic_structure
@pytest.mark.parametrize(
    "prune_by_energy, _expected_pruning_mask",
    [
        (None, [True, True, True, True, True, True, True]),
        ((40, "kj/mol"), [True, True, True, True, True, True, False]),
        ((5, "kj/mol"), [True, False, True, True, False, True, False]),
    ],
)
def test_calculate_electronic_structure5(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    prune_by_energy: Optional[Tuple[Union[int, float], str]],
    _expected_pruning_mask: List[bool],
) -> None:
    """Test for the ``calculate_electronic_structure()`` method: valid input with energy
    pruning.
    """
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name="clopidogrel.xyz")
    f.read_input(
        input_value=input_string,
        namespace="irrelevant",
        input_format="file",
        output_directory="irrelevant_out_dir",
    )
    f.set_charge(charge=0)
    f.set_multiplicity(multiplicity=1)
    assert f.mol_vault is not None
    assert f.mol_vault.is_valid == [True] * 7
    assert f.mol_vault.energies_n == []
    assert f.mol_vault.energies_n_read is False
    assert f.mol_vault.electronic_strucs_n == []
    assert f.mol_vault.electronic_strucs_n_minus1 == []
    assert f.mol_vault.electronic_strucs_n_plus1 == []
    assert f.mol_vault.electronic_struc_types_n == []
    assert f.mol_vault.electronic_struc_types_n_minus1 == []
    assert f.mol_vault.electronic_struc_types_n_plus1 == []

    # Mock the actual single-point energy calculation and the generated electronic structure file
    _mock_el_struc_file = os.path.join(
        os.getcwd(), "irrelevant_out_dir", "dummy_el_struc_file.molden"
    )
    with open(_mock_el_struc_file, "w") as file:
        file.write("Nothing to see here.\n")

    _mocked_energies = [
        (-123.123, "kj_mol"),
        (-112.123, "kj_mol"),
        (-123.123, "kj_mol"),
        (-123.123, "kj_mol"),
        (-100.123, "kj_mol"),
        (-123.123, "kj_mol"),
        (-45.123, "kj_mol"),
    ]
    _mocked_energies_read = True
    _mocked_electronic_strucs = [_mock_el_struc_file] * 7
    _expected_electronic_struc_types = ["molden"] * 7

    assert f.mol_vault.is_valid == [True] * 7

    mock = mocker.patch(
        "bonafide.utils.sp_xtb.XtbSP.run",
        return_value=(_mocked_energies, _mocked_electronic_strucs),
    )

    # Calculate electronic structure
    f.calculate_electronic_structure(engine="xtb", redox="n", prune_by_energy=prune_by_energy)

    # Check mol vault
    assert f.mol_vault.is_valid == _expected_pruning_mask

    mock.assert_called_with(state="n", write_el_struc_file=True)
    assert mock.call_count == 1

    assert f.mol_vault.energies_n == _mocked_energies
    assert f.mol_vault.energies_n_read is _mocked_energies_read
    assert f.mol_vault.electronic_strucs_n == _mocked_electronic_strucs
    assert f.mol_vault.electronic_strucs_n_minus1 == []
    assert f.mol_vault.electronic_strucs_n_plus1 == []
    assert f.mol_vault.electronic_struc_types_n == _expected_electronic_struc_types
    assert f.mol_vault.electronic_struc_types_n_minus1 == []
    assert f.mol_vault.electronic_struc_types_n_plus1 == []

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)

    # Clean up
    shutil.rmtree(path="irrelevant_out_dir")
    clean_up_logfile()


def _vault_testing(
    vault: MolVault,
    _preattach_to_state: str,
    _preattach_what: str,
    _file_path: List[str],
    _energies: List[Tuple[float, str]],
):
    """Helper function for testing the molecule vault in test_calculate_electronic_structure6."""
    if _preattach_to_state == "n" and _preattach_what == "el_struc":
        assert vault.energies_n == []
        assert vault.energies_n_minus1 == []
        assert vault.energies_n_plus1 == []
        assert vault.energies_n_read is False
        assert vault.energies_n_minus1_read is False
        assert vault.energies_n_plus1_read is False
        assert vault.electronic_strucs_n == _file_path
        assert vault.electronic_strucs_n_minus1 == []
        assert vault.electronic_strucs_n_plus1 == []
        assert vault.electronic_struc_types_n == ["molden"] * 7
        assert vault.electronic_struc_types_n_minus1 == []
        assert vault.electronic_struc_types_n_plus1 == []

    elif _preattach_to_state == "n" and _preattach_what == "energies":
        assert vault.energies_n == _energies
        assert vault.energies_n_minus1 == []
        assert vault.energies_n_plus1 == []
        assert vault.energies_n_read is True
        assert vault.energies_n_minus1_read is False
        assert vault.energies_n_plus1_read is False
        assert vault.electronic_strucs_n == []
        assert vault.electronic_strucs_n_minus1 == []
        assert vault.electronic_strucs_n_plus1 == []
        assert vault.electronic_struc_types_n == []
        assert vault.electronic_struc_types_n_minus1 == []
        assert vault.electronic_struc_types_n_plus1 == []

    elif _preattach_to_state == "n-1" and _preattach_what == "el_struc":
        assert vault.energies_n == []
        assert vault.energies_n_minus1 == []
        assert vault.energies_n_plus1 == []
        assert vault.energies_n_read is False
        assert vault.energies_n_minus1_read is False
        assert vault.energies_n_plus1_read is False
        assert vault.electronic_strucs_n == []
        assert vault.electronic_strucs_n_minus1 == _file_path
        assert vault.electronic_strucs_n_plus1 == []
        assert vault.electronic_struc_types_n == []
        assert vault.electronic_struc_types_n_minus1 == ["molden"] * 7
        assert vault.electronic_struc_types_n_plus1 == []

    elif _preattach_to_state == "n-1" and _preattach_what == "energies":
        assert vault.energies_n == []
        assert vault.energies_n_minus1 == _energies
        assert vault.energies_n_plus1 == []
        assert vault.energies_n_read is False
        assert vault.energies_n_minus1_read is True
        assert vault.energies_n_plus1_read is False
        assert vault.electronic_strucs_n == []
        assert vault.electronic_strucs_n_minus1 == []
        assert vault.electronic_strucs_n_plus1 == []
        assert vault.electronic_struc_types_n == []
        assert vault.electronic_struc_types_n_minus1 == []
        assert vault.electronic_struc_types_n_plus1 == []

    elif _preattach_to_state == "n+1" and _preattach_what == "el_struc":
        assert vault.energies_n == []
        assert vault.energies_n_minus1 == []
        assert vault.energies_n_plus1 == []
        assert vault.energies_n_read is False
        assert vault.energies_n_minus1_read is False
        assert vault.energies_n_plus1_read is False
        assert vault.electronic_strucs_n == []
        assert vault.electronic_strucs_n_minus1 == []
        assert vault.electronic_strucs_n_plus1 == _file_path
        assert vault.electronic_struc_types_n == []
        assert vault.electronic_struc_types_n_minus1 == []
        assert vault.electronic_struc_types_n_plus1 == ["molden"] * 7

    elif _preattach_to_state == "n+1" and _preattach_what == "energies":
        assert vault.energies_n == []
        assert vault.energies_n_minus1 == []
        assert vault.energies_n_plus1 == _energies
        assert vault.energies_n_read is False
        assert vault.energies_n_minus1_read is False
        assert vault.energies_n_plus1_read is True
        assert vault.electronic_strucs_n == []
        assert vault.electronic_strucs_n_minus1 == []
        assert vault.electronic_strucs_n_plus1 == []
        assert vault.electronic_struc_types_n == []
        assert vault.electronic_struc_types_n_minus1 == []
        assert vault.electronic_struc_types_n_plus1 == []

    else:
        raise ValueError(
            f"Unexpected pre-attachment state '{_preattach_to_state}' encountered in test."
        )


@pytest.mark.calculate_electronic_structure
@pytest.mark.parametrize(
    "engine_name, redox_state, _preattach_to_state, _preattach_what",
    [
        ("xtb", "n", "n", "el_struc"),
        ("psi4", "n", "n", "el_struc"),
        ("xtb", "n", "n", "energies"),
        ("psi4", "n", "n", "energies"),
        ("xtb", "all", "n", "el_struc"),
        ("psi4", "all", "n", "el_struc"),
        ("xtb", "all", "n", "energies"),
        ("psi4", "all", "n", "energies"),
        ("xtb", "n-1", "n-1", "el_struc"),
        ("psi4", "n-1", "n-1", "el_struc"),
        ("xtb", "n-1", "n-1", "energies"),
        ("psi4", "n-1", "n-1", "energies"),
        ("xtb", "n+1", "n-1", "el_struc"),
        ("psi4", "n+1", "n-1", "el_struc"),
        ("xtb", "n+1", "n-1", "energies"),
        ("psi4", "n+1", "n-1", "energies"),
        ("xtb", "n+1", "n+1", "el_struc"),
        ("psi4", "n+1", "n+1", "el_struc"),
        ("xtb", "n+1", "n+1", "energies"),
        ("psi4", "n+1", "n+1", "energies"),
    ],
)
def test_calculate_electronic_structure6(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    engine_name: str,
    redox_state: str,
    _preattach_to_state: str,
    _preattach_what: str,
) -> None:
    """Test for the ``calculate_electronic_structure()`` method: not possible because data
    is already in the vault."""

    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name="clopidogrel.xyz")
    f.read_input(
        input_value=input_string,
        namespace="irrelevant",
        input_format="file",
        output_directory="irrelevant_out_dir",
    )
    f.set_charge(charge=0)
    f.set_multiplicity(multiplicity=1)

    if _preattach_what == "el_struc":
        _preattched_file_path = fetch_data_file(file_name="dummy_electronic_struc_file_01.molden")
        _preattched_file_path = [_preattched_file_path] * 7
        f.attach_electronic_structure(
            electronic_structure_data=_preattched_file_path, state=_preattach_to_state
        )
    elif _preattach_what == "energies":
        _preattached_energies = [(-42.42, "kj_mol")] * 7
        f.attach_energy(energy_data=_preattached_energies, state=_preattach_to_state)
    else:
        raise ValueError(f"Unexpected pre-attachment type '{_preattach_what}' encountered in test.")

    # Check mol vault pre-attachment
    assert f.mol_vault is not None
    assert f.mol_vault.is_valid == [True] * 7
    _vault_testing(
        vault=f.mol_vault,
        _preattach_to_state=_preattach_to_state,
        _preattach_what=_preattach_what,
        _file_path=_preattched_file_path if _preattach_what == "el_struc" else [],
        _energies=_preattached_energies if _preattach_what == "energies" else [],
    )

    # Mock the actual single-point energy calculation and the generated electronic structure file
    _mock_el_struc_file = os.path.join(
        os.getcwd(), "irrelevant_out_dir", "dummy_el_struc_file.molden"
    )
    with open(_mock_el_struc_file, "w") as file:
        file.write("Nothing to see here.\n")

    _mocked_energies = [
        (-123.123, "kj_mol"),
        (-112.123, "kj_mol"),
        (-123.123, "kj_mol"),
        (-123.123, "kj_mol"),
        (-100.123, "kj_mol"),
        (-123.123, "kj_mol"),
        (-45.123, "kj_mol"),
    ]
    _mocked_electronic_strucs = [_mock_el_struc_file] * 7
    _expected_electronic_struc_types = ["molden"] * 7

    if engine_name.strip().lower() == "psi4":
        _patch_path = "bonafide.utils.sp_psi4.Psi4SP.run"
    if engine_name.strip().lower() == "xtb":
        _patch_path = "bonafide.utils.sp_xtb.XtbSP.run"
    mock = mocker.patch(
        _patch_path,
        return_value=(_mocked_energies, _mocked_electronic_strucs),
    )

    # Calculate electronic structure
    f.calculate_electronic_structure(engine=engine_name, redox=redox_state)

    # Check mol vault
    assert f.mol_vault.is_valid == [True] * 7

    if redox_state == "n" and _preattach_to_state == "n":
        assert mock.call_count == 0
        _vault_testing(
            vault=f.mol_vault,
            _preattach_to_state=_preattach_to_state,
            _preattach_what=_preattach_what,
            _file_path=_preattched_file_path if _preattach_what == "el_struc" else [],
            _energies=_preattached_energies if _preattach_what == "energies" else [],
        )
    elif redox_state == _preattach_to_state:
        assert mock.call_count == 1

        assert f.mol_vault.energies_n == _mocked_energies
        assert f.mol_vault.energies_n_read is True
        assert f.mol_vault.electronic_strucs_n == _mocked_electronic_strucs
        assert f.mol_vault.electronic_struc_types_n == _expected_electronic_struc_types

    elif redox_state == "all" and _preattach_to_state == "n":
        assert mock.call_count == 2

        assert f.mol_vault.energies_n_minus1 == _mocked_energies
        assert f.mol_vault.energies_n_plus1 == _mocked_energies
        assert f.mol_vault.energies_n_minus1_read is True
        assert f.mol_vault.energies_n_plus1_read is True
        assert f.mol_vault.electronic_strucs_n_minus1 == _mocked_electronic_strucs
        assert f.mol_vault.electronic_strucs_n_plus1 == _mocked_electronic_strucs
        assert f.mol_vault.electronic_struc_types_n_minus1 == _expected_electronic_struc_types
        assert f.mol_vault.electronic_struc_types_n_plus1 == _expected_electronic_struc_types

    elif redox_state == "n+1" and _preattach_to_state == "n-1":
        assert mock.call_count == 2

        assert f.mol_vault.energies_n == _mocked_energies
        assert f.mol_vault.energies_n_plus1 == _mocked_energies
        assert f.mol_vault.energies_n_read is True
        assert f.mol_vault.energies_n_plus1_read is True
        assert f.mol_vault.electronic_strucs_n == _mocked_electronic_strucs
        assert f.mol_vault.electronic_strucs_n_plus1 == _mocked_electronic_strucs
        assert f.mol_vault.electronic_struc_types_n == _expected_electronic_struc_types
        assert f.mol_vault.electronic_struc_types_n_plus1 == _expected_electronic_struc_types

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno != logging.ERROR for record in caplog.records)

    # Clean up
    shutil.rmtree(path="irrelevant_out_dir")
    clean_up_logfile()


@pytest.mark.calculate_electronic_structure
@pytest.mark.parametrize(
    "input_file, charge, multiplicity, redox_state, engine_name, options, _expected_energies",
    [
        (
            "clopidogrel-conf_01.xyz",
            0,
            1,
            "n",
            "xtb",
            [],
            [-160865.73613247764],
        ),
        (
            "clopidogrel-conf_01.xyz",
            0,
            1,
            "n",
            "psi4",
            [("psi4.basis", "sto-3g")],
            [-4366830.812556853],
        ),
        (
            "spiro_mol_e.sdf",
            0,
            1,
            "n",
            "xtb",
            [],
            [
                -157976.40992410545,
                -157976.80091278197,
                -157964.76736591622,
                -157965.26376009884,
                -157961.54100743012,
            ],
        ),
    ],
)
def test_calculate_electronic_structure7(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    input_file: str,
    charge: int,
    multiplicity: int,
    redox_state: str,
    engine_name: str,
    options: List[Tuple[str, Any]],
    _expected_energies: List[float],
) -> None:
    """Test for the ``calculate_electronic_structure()`` method: end-to-end testing without
    patching.
    """
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name=input_file)
    f.read_input(
        input_value=input_string,
        namespace="irrelevant",
        input_format="file",
        output_directory="irrelevant_out_dir",
    )
    f.set_charge(charge=charge)
    f.set_multiplicity(multiplicity=multiplicity)

    if options != []:
        f.set_options(configs=options)

    assert f.mol_vault is not None

    # Calculate electronic structure
    f.calculate_electronic_structure(engine=engine_name, redox=redox_state)

    # Check molecule vault
    if redox_state == "n":
        assert f.mol_vault.energies_n_read is True
        assert len(f.mol_vault.energies_n) == len(_expected_energies)
        for calc_energy, expected_energy in zip(f.mol_vault.energies_n, _expected_energies):
            assert calc_energy[0] == pytest.approx(expected_energy)

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)

    # Clean up
    shutil.rmtree(path="irrelevant_out_dir")
    clean_up_logfile()
