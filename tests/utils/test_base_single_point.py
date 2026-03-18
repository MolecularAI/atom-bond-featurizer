"""Test functions for the ``bonafide.utils.base_single_point`` module."""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import pytest
from rdkit import Chem

from bonafide.utils.base_single_point import BaseSinglePoint
from bonafide.utils.io_ import read_xyz_file

if TYPE_CHECKING:
    from bonafide.utils.molecule_vault import MolVault

###########################################
# Helper classes for the individual tests #
###########################################


class SPClass(BaseSinglePoint):
    """Valid single-point energy calculation class with dummy calculate method."""

    sp_calculate_counter = 0

    def __init__(self, **kwargs) -> None:
        self.engine_name = "dummy_engine"
        super().__init__(**kwargs)

    def calculate(self, write_el_struc_file: bool) -> Tuple[float, Optional[str]]:
        # Increment global counter to verify that this method was called
        self.sp_calculate_counter += 1

        if write_el_struc_file is True:
            return -12579.548, "dummy/file/path/el_struc.molden"
        return -52579.548, None


class SPClassFailure(BaseSinglePoint):
    """Valid single-point energy calculation but with fail in calculate."""

    sp_calculate_counter = 0

    def __init__(self, **kwargs) -> None:
        self.engine_name = "dummy_engine"
        super().__init__(**kwargs)

    def calculate(self, write_el_struc_file: bool) -> Tuple[float, Optional[str]]:
        # Increment global counter to verify that this method was called
        self.sp_calculate_counter += 1

        if self.sp_calculate_counter == 2 or self.sp_calculate_counter == 5:
            raise RuntimeError("Simulated failure in calculate() method.")

        if write_el_struc_file is True:
            return -12579.548, "dummy/file/path/el_struc.molden"
        return -52579.548, None


class SPClassNoCalculate(BaseSinglePoint):
    """Invalid single-point energy calculation class with missing calculate method."""

    def __init__(self) -> None:
        self.engine_name = "dummy_engine"
        super().__init__()


class SPClassNoEngineName(BaseSinglePoint):
    """Invalid single-point energy calculation class with missing engine_name attribute."""

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        pass


class DummyMolVault:
    """Dummy MolVault class."""

    def __init__(self) -> None:
        self.namespace = "dummy_namespace"


####################################
# Tests for the __init__() method. #
####################################


@pytest.mark.base_single_point_init
def test___init__(caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``__init__()`` method: valid example."""
    test_kwargs = {
        "charge": -1,
        "multiplicity": 2,
        "mol_vault": DummyMolVault(),
        "custom_attr": "test_value",
    }
    sp = SPClass(**test_kwargs)

    assert sp.charge == -1
    assert sp.multiplicity == 2
    assert sp.mol_vault.namespace == "dummy_namespace"
    assert sp.custom_attr == "test_value"
    assert sp.engine_name == "dummy_engine"

    # Check logging
    assert len(caplog.records) == 0


@pytest.mark.base_single_point_init
def test___init__2(caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``__init__()`` method: missing calculate() method."""
    with pytest.raises(
        NotImplementedError,
        match=r"calculate\(\) method must be implemented in engine-specific single-point energy "
        "class",
    ):
        SPClassNoCalculate()

    # Check logging
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)


@pytest.mark.base_single_point_init
def test___init__3(caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``__init__()`` method: missing engine_name attribute."""
    with pytest.raises(
        AttributeError,
        match="Attribute 'engine_name' must be set in engine-specific single-point energy class",
    ):
        SPClassNoEngineName()

    # Check logging
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)


###############################
# Tests for the run() method. #
###############################


@pytest.mark.base_single_point_run
@pytest.mark.parametrize(
    "input_file_name, state, write_el_struc_file, _is_valid_list",
    [
        ("fluoromethanol.xyz", "n", True, [True]),
        ("fluoromethanol.xyz", "n", True, [False]),
        ("fluoromethanol.xyz", "n+1", True, [False]),
        ("fluoromethanol.xyz", "n-1", True, [True]),
        ("clopidogrel.xyz", "n", True, [True, True, True, True, True, True, True]),
        ("clopidogrel.xyz", "n", True, [True, False, True, True, False, True, True]),
        ("clopidogrel.xyz", "n", True, [False, False, False, False, False, False, False]),
        ("clopidogrel.xyz", "n+1", True, [True, True, True, True, True, True, True]),
        ("clopidogrel.xyz", "n-1", True, [True, False, True, True, False, True, True]),
        ("clopidogrel.xyz", "n-1", True, [False, False, False, False, False, False, False]),
        ("fluoromethanol.xyz", "n", False, [True]),
        ("fluoromethanol.xyz", "n", False, [False]),
        ("fluoromethanol.xyz", "n+1", False, [False]),
        ("fluoromethanol.xyz", "n-1", False, [True]),
        ("clopidogrel.xyz", "n", False, [True, True, True, True, True, True, True]),
        ("clopidogrel.xyz", "n", False, [True, False, True, True, False, True, True]),
        ("clopidogrel.xyz", "n", False, [False, False, False, False, False, False, False]),
        ("clopidogrel.xyz", "n+1", False, [True, True, True, True, True, True, True]),
        ("clopidogrel.xyz", "n-1", False, [True, False, True, True, False, True, True]),
        ("clopidogrel.xyz", "n-1", False, [False, False, False, False, False, False, False]),
    ],
)
def test_run(
    caplog: pytest.LogCaptureFixture,
    fetch_data_file: Callable[[str], str],
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
    input_file_name: str,
    state: str,
    write_el_struc_file: bool,
    _is_valid_list: List[bool],
) -> None:
    """Test for the ``run()`` method: valid examples."""
    # Preprocess input file
    input_file = fetch_data_file(file_name=input_file_name)
    mol_inputs, error_message = read_xyz_file(file_path=input_file)
    assert error_message is None

    # Arrange mol vault
    mol_vault = fresh_mol_vault(mol_inputs=mol_inputs, namespace="irrelevant", input_type="xyz")
    mol_vault.initialize_mol()
    mol_vault.get_elements()
    mol_vault.charge = "irrelevant"
    mol_vault.multiplicity = "irrelevant"
    mol_vault.is_valid = _is_valid_list
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])

    sp_kwargs = {"mol_vault": mol_vault, "_keep_output_files": True}
    dummy = SPClass(**sp_kwargs)
    energies, electronic_strucs = dummy.run(state=state, write_el_struc_file=write_el_struc_file)

    assert len(energies) == len(_is_valid_list)
    assert len(electronic_strucs) == len(_is_valid_list)

    if state == "n":
        _expected_energies = []
        _expected_electronic_strucs = []
        for v in _is_valid_list:
            if v is True and write_el_struc_file is True:
                _expected_energies.append((-12579.548, "kj_mol"))
                _expected_electronic_strucs.append("dummy/file/path/el_struc.molden")
            elif v is True and write_el_struc_file is False:
                _expected_energies.append((-52579.548, "kj_mol"))
                _expected_electronic_strucs.append(None)
            else:
                _expected_energies.append((None, "kj_mol"))
                _expected_electronic_strucs.append(None)

        assert energies == _expected_energies
        assert electronic_strucs == _expected_electronic_strucs
        assert dummy.sp_calculate_counter == Counter(_is_valid_list)[True]

    else:
        assert dummy.sp_calculate_counter == len(_is_valid_list)

    # Check logging
    assert len(caplog.records) > 0
    assert all(record.levelno != logging.ERROR for record in caplog.records)


@pytest.mark.base_single_point_run
def test_run2(
    caplog: pytest.LogCaptureFixture,
    fetch_data_file: Callable[[str], str],
    fresh_mol_vault: Callable[[List[str], str, str], MolVault],
) -> None:
    """Test for the ``run()`` method: failure in calculate."""
    # Preprocess input file
    input_file = fetch_data_file(file_name="clopidogrel.xyz")
    mol_inputs, error_message = read_xyz_file(file_path=input_file)
    assert error_message is None

    # Arrange mol vault
    mol_vault = fresh_mol_vault(mol_inputs=mol_inputs, namespace="irrelevant", input_type="xyz")
    mol_vault.initialize_mol()
    mol_vault.get_elements()
    mol_vault.charge = "irrelevant"
    mol_vault.multiplicity = "irrelevant"
    mol_vault.is_valid = [True, True, True, True, True, True, False]
    assert all([isinstance(mol, Chem.rdchem.Mol) for mol in mol_vault.mol_objects])

    sp_kwargs = {"mol_vault": mol_vault, "_keep_output_files": True}
    dummy = SPClassFailure(**sp_kwargs)
    energies, electronic_strucs = dummy.run(state="n", write_el_struc_file=True)

    assert len(energies) == 7
    assert len(electronic_strucs) == 7
    assert energies == [
        (-12579.548, "kj_mol"),
        (None, "kj_mol"),
        (-12579.548, "kj_mol"),
        (-12579.548, "kj_mol"),
        (None, "kj_mol"),
        (-12579.548, "kj_mol"),
        (None, "kj_mol"),
    ]
    assert electronic_strucs == [
        "dummy/file/path/el_struc.molden",
        None,
        "dummy/file/path/el_struc.molden",
        "dummy/file/path/el_struc.molden",
        None,
        "dummy/file/path/el_struc.molden",
        None,
    ]
    assert dummy.sp_calculate_counter == 6

    # Check logging
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)
