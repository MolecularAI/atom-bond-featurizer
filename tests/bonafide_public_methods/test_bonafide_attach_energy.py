"""Test functions for the ``attach_energy()`` method in the ``bonafide.bonafide`` module."""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import pytest
from rdkit import RDLogger

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


@pytest.mark.attach_energy
@pytest.mark.parametrize(
    "input_string, energy_data, state, prune_by_energy, _expexted_processed_energies, "
    "_expected_processed_input_energies",
    [
        ("clopidogrel-conf_01.xyz", (123.123, "EH"), "n", None, [323259.39217572], ["eh"]),
        (
            "clopidogrel-conf_01.xyz",
            [(123.123, "kilojoules-per-mol")],
            "n",
            None,
            [123.123],
            ["kj_mol"],
        ),
        ("clopidogrel-conf_01.xyz", (123.123, "EH"), "n+1", None, [323259.39217572], ["eh"]),
        ("clopidogrel-conf_01.xyz", (-123.123, "EH"), "n-1", None, [-323259.39217572], ["eh"]),
        (
            "radical_cation_e.sdf",
            [(123.123, "kj_mol"), (456.456, "EH")],
            "n",
            None,
            [123.123, 1198425.0636758402],
            ["kj_mol", "eh"],
        ),
        (
            "radical_cation_e.sdf",
            [(123.123, "kj_mol"), (456.456, "kcalpermole")],
            "n-1",
            (5, "kcal/mol"),
            [123.123, 1909.8119040],
            ["kj_mol", "kcal_mol"],
        ),
        (
            "radical_cation_e.sdf",
            [(3, "kj_mol"), (-456.456, "EH")],
            "n+1",
            None,
            [3, -1198425.0636758402],
            ["kj_mol", "eh"],
        ),
    ],
)
def test_attach_energy(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    input_string: str,
    energy_data: Union[Tuple[Union[float, int], str], List[Tuple[Union[float, int], str]]],
    state: str,
    prune_by_energy: Optional[Tuple[Union[float, int], str]],
    _expexted_processed_energies: List[float],
    _expected_processed_input_energies: List[str],
) -> None:
    """Test for the ``attach_energy()`` method: valid examples."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    input_string = fetch_data_file(file_name=input_string)
    f.read_input(
        input_value=input_string,
        namespace="irrelevant",
        input_format="file",
        output_directory="irrelevant_out_dir",
    )
    assert f.mol_vault is not None
    assert f.mol_vault.energies_n == []
    assert f.mol_vault.energies_n_minus1 == []
    assert f.mol_vault.energies_n_plus1 == []
    assert f.mol_vault.energies_n_read is False
    assert f.mol_vault.energies_n_minus1_read is False
    assert f.mol_vault.energies_n_plus1_read is False
    assert f.mol_vault._input_energies_n == []
    assert f.mol_vault._input_energies_n_minus1 == []
    assert f.mol_vault._input_energies_n_plus1 == []

    # Attach energy data
    f.attach_energy(energy_data=energy_data, state=state, prune_by_energy=prune_by_energy)

    # Check attached energies
    if type(energy_data) == tuple:
        energy_data = [energy_data]

    if state == "n":
        assert f.mol_vault.energies_n_read is True
        assert len(f.mol_vault.energies_n) == len(_expexted_processed_energies)
        assert len(f.mol_vault._input_energies_n) == len(_expexted_processed_energies)

        for (energy, unit), expected_energy in zip(
            f.mol_vault.energies_n, _expexted_processed_energies
        ):
            assert pytest.approx(energy) == expected_energy
            assert unit == "kj_mol"

        for (energy, unit), (idx, (inp_energy, _)) in zip(
            f.mol_vault._input_energies_n, enumerate(energy_data)
        ):
            assert pytest.approx(energy) == inp_energy
            assert unit == _expected_processed_input_energies[idx]

    if state == "n-1":
        assert f.mol_vault.energies_n_minus1_read is True
        assert len(f.mol_vault.energies_n_minus1) == len(_expexted_processed_energies)
        assert len(f.mol_vault._input_energies_n_minus1) == len(_expexted_processed_energies)

        for (energy, unit), expected_energy in zip(
            f.mol_vault.energies_n_minus1, _expexted_processed_energies
        ):
            assert pytest.approx(energy) == expected_energy
            assert unit == "kj_mol"

        for (energy, unit), (idx, (inp_energy, _)) in zip(
            f.mol_vault._input_energies_n_minus1, enumerate(energy_data)
        ):
            assert pytest.approx(energy) == inp_energy
            assert unit == _expected_processed_input_energies[idx]

    if state == "n+1":
        assert f.mol_vault.energies_n_plus1_read is True
        assert len(f.mol_vault.energies_n_plus1) == len(_expexted_processed_energies)
        assert len(f.mol_vault._input_energies_n_plus1) == len(_expexted_processed_energies)

        for (energy, unit), expected_energy in zip(
            f.mol_vault.energies_n_plus1, _expexted_processed_energies
        ):
            assert pytest.approx(energy) == expected_energy
            assert unit == "kj_mol"

        for (energy, unit), (idx, (inp_energy, _)) in zip(
            f.mol_vault._input_energies_n_plus1, enumerate(energy_data)
        ):
            assert pytest.approx(energy) == inp_energy
            assert unit == _expected_processed_input_energies[idx]

    # Check logs
    assert len(caplog.records) > 0
    if prune_by_energy is None:
        assert all(record.levelno == logging.INFO for record in caplog.records)
    else:
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    # Clean up
    os.rmdir(path="irrelevant_out_dir")
    clean_up_logfile()


@pytest.mark.attach_energy
def test_attach_energy2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``attach_energy()`` method: fails because no molecule read yet."""
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Attach energy data
    with pytest.raises(ValueError, match="attaching energy data"):
        f.attach_energy(energy_data="irrelevant")
    assert f.mol_vault is None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.attach_energy
def test_attach_energy3(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``attach_energy()`` method: fails because molecule in vault is 2D."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    f.read_input(input_value="O(C([H])([H])F)[H]", namespace="irrelevant")
    assert f.mol_vault is not None

    # Attach enery data
    with pytest.raises(
        ValueError,
        match="Attaching molecular energy values to a molecule vault hosting a 2D molecule is "
        "not allowed.",
    ):
        f.attach_energy(energy_data="irrelevant")
    assert f.mol_vault is not None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.attach_energy
@pytest.mark.parametrize(
    "energy_data, state, prune_by_energy, _preattach, _error_type",
    [
        (False, "n", None, False, TypeError),
        ([[1.25, "kj_mol"]], "n", None, False, TypeError),
        ([(1.25, "kj_mol"), (1.25,)], "n", None, False, ValueError),
        ([(1.25, "kj_mol"), (3.25, "kj_mol")], ["n_plus1"], None, False, TypeError),
        ([(1.25, "kj_mol"), (3.25, "kj_mol")], "n-1", None, False, ValueError),
        ((1.25, "kj_mol"), "n", None, True, ValueError),
        (("1.25", "kj_mol"), "n", None, False, TypeError),
        ((1.25, None), "n", None, False, TypeError),
        ((1.25, "i-am-not-a-unit"), "n+1", None, False, ValueError),
    ],
)
def test_attach_energy4(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    energy_data: Any,
    state: Any,
    prune_by_energy: Any,
    _preattach: bool,
    _error_type: Any,
) -> None:
    """Test for the ``attach_energy()`` method: invalid inputs."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    _file_path = fetch_data_file(file_name="clopidogrel-conf_01.xyz")
    f.read_input(input_value=_file_path, namespace="irrelevant", input_format="file")
    assert f.mol_vault is not None

    # Preattach energy data if requested
    if _preattach is True:
        f.attach_energy(energy_data=(1, "kj_mol"))

    # Attach energy data
    with pytest.raises(_error_type):
        f.attach_energy(energy_data=energy_data, state=state, prune_by_energy=prune_by_energy)

    if _preattach is False:
        assert f.mol_vault.energies_n == []
        assert f.mol_vault.energies_n_minus1 == []
        assert f.mol_vault.energies_n_plus1 == []

        assert f.mol_vault.energies_n_read is False
        assert f.mol_vault.energies_n_minus1_read is False
        assert f.mol_vault.energies_n_plus1_read is False

    else:
        assert f.mol_vault.energies_n == [(1.0, "kj_mol")]
        assert f.mol_vault.energies_n_minus1 == []
        assert f.mol_vault.energies_n_plus1 == []

        assert f.mol_vault.energies_n_read is True
        assert f.mol_vault.energies_n_minus1_read is False
        assert f.mol_vault.energies_n_plus1_read is False

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.attach_energy
@pytest.mark.parametrize(
    "energy_data, prune_by_energy, _expected_pruning_mask",
    [
        (
            [
                (0, "kj/mol"),
                (10, "kj_mol"),
                (4, "kj_mol"),
                (-1.1, "kj_mol"),
                (6, "kj_mol"),
                (13, "kj_mol"),
                (-4, "kj-mol"),
            ],
            (1, "kcal/mol"),
            [True, False, False, True, False, False, True],
        )
    ],
)
def test_attach_energy5(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    energy_data: List[Tuple[Union[float, int], str]],
    prune_by_energy: Tuple[Union[float, int], str],
    _expected_pruning_mask: List[bool],
) -> None:
    """Test for the ``attach_energy()`` method: energy pruning."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    _file_path = fetch_data_file(file_name="clopidogrel_e_one_missing.xyz")
    f.read_input(
        input_value=_file_path,
        namespace="irrelevant",
        input_format="file",
        output_directory="irrelevant_out_dir",
    )
    assert f.mol_vault is not None
    assert f.mol_vault.energies_n == []
    assert f.mol_vault.energies_n_read is False
    assert f.mol_vault._input_energies_n == []

    # Attach energy data
    f.attach_energy(energy_data=energy_data, prune_by_energy=prune_by_energy)

    energy_data = [(float(energy), "kj_mol") for energy, _ in energy_data]

    # Check mol vault
    assert f.mol_vault.energies_n == energy_data
    assert f.mol_vault.energies_n_read is True
    assert f.mol_vault._input_energies_n == energy_data

    assert f.mol_vault.is_valid == _expected_pruning_mask

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)

    # Clean up
    os.rmdir(path="irrelevant_out_dir")
    clean_up_logfile()
