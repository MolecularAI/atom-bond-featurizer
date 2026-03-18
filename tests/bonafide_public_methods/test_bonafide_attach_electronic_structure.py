"""Test functions for the ``attach_electronic_structure()`` method in the ``bonafide.bonafide``
module.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, List, Union

import pytest
from rdkit import RDLogger

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


@pytest.mark.attach_electronic_structure
@pytest.mark.parametrize(
    "input_string, electronic_structure_data, state",
    [
        ("clopidogrel-conf_01.xyz", "dummy_electronic_struc_file_01.molden", "n"),
        ("clopidogrel-conf_01.xyz", ["dummy_electronic_struc_file_01.molden"], "n"),
        ("clopidogrel-conf_01.xyz", "dummy_electronic_struc_file_01.molden", "n+1"),
        ("clopidogrel-conf_01.xyz", "dummy_electronic_struc_file_01.molden", "n-1"),
        (
            "radical_cation_e.sdf",
            ["dummy_electronic_struc_file_01.molden", "dummy_electronic_struc_file_02.molden"],
            "n",
        ),
        (
            "radical_cation_e.sdf",
            ["dummy_electronic_struc_file_01.molden", "dummy_electronic_struc_file_02.molden"],
            "n-1",
        ),
        (
            "radical_cation_e.sdf",
            ["dummy_electronic_struc_file_01.molden", "dummy_electronic_struc_file_02.molden"],
            "n+1",
        ),
    ],
)
def test_attach_electronic_structure(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    input_string: str,
    electronic_structure_data: Union[str, List[str]],
    state: str,
) -> None:
    """Test for the ``attach_electronic_structure()`` method: valid examples."""
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
    assert f.mol_vault.electronic_strucs_n == []
    assert f.mol_vault.electronic_strucs_n_minus1 == []
    assert f.mol_vault.electronic_strucs_n_plus1 == []
    assert f.mol_vault.electronic_struc_types_n == []
    assert f.mol_vault.electronic_struc_types_n_minus1 == []
    assert f.mol_vault.electronic_struc_types_n_plus1 == []

    # Format file path
    if type(electronic_structure_data) == list:
        electronic_structure_data = [
            fetch_data_file(file_name=fn) for fn in electronic_structure_data
        ]
        _ref_list = electronic_structure_data
    else:
        electronic_structure_data = fetch_data_file(file_name=electronic_structure_data)
        _ref_list = [electronic_structure_data]

    # Attach electronic structure data
    f.attach_electronic_structure(electronic_structure_data=electronic_structure_data, state=state)

    if state == "n":
        assert f.mol_vault.electronic_strucs_n == _ref_list
        assert f.mol_vault.electronic_strucs_n_minus1 == []
        assert f.mol_vault.electronic_strucs_n_plus1 == []

        assert f.mol_vault.electronic_struc_types_n == ["molden"] * len(_ref_list)
        assert f.mol_vault.electronic_struc_types_n_minus1 == []
        assert f.mol_vault.electronic_struc_types_n_plus1 == []

    if state == "n-1":
        assert f.mol_vault.electronic_strucs_n == []
        assert f.mol_vault.electronic_strucs_n_minus1 == _ref_list
        assert f.mol_vault.electronic_strucs_n_plus1 == []

        assert f.mol_vault.electronic_struc_types_n == []
        assert f.mol_vault.electronic_struc_types_n_minus1 == ["molden"] * len(_ref_list)
        assert f.mol_vault.electronic_struc_types_n_plus1 == []

    if state == "n+1":
        assert f.mol_vault.electronic_strucs_n == []
        assert f.mol_vault.electronic_strucs_n_minus1 == []
        assert f.mol_vault.electronic_strucs_n_plus1 == _ref_list

        assert f.mol_vault.electronic_struc_types_n == []
        assert f.mol_vault.electronic_struc_types_n_minus1 == []
        assert f.mol_vault.electronic_struc_types_n_plus1 == ["molden"] * len(_ref_list)

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)

    # Clean up
    os.rmdir(path="irrelevant_out_dir")
    clean_up_logfile()


@pytest.mark.attach_electronic_structure
def test_attach_electronic_structure2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``attach_electronic_structure()`` method: fails because no molecule read
    yet.
    """
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Attach electronic structure data
    with pytest.raises(ValueError, match="attaching electronic structure data"):
        f.attach_electronic_structure(electronic_structure_data="irrelevant", state="n")
    assert f.mol_vault is None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.attach_electronic_structure
def test_attach_electronic_structure3(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``attach_electronic_structure()`` method: fails because molecule in vault
    is 2D.
    """
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    f.read_input(input_value="O(C([H])([H])F)[H]", namespace="irrelevant")
    assert f.mol_vault is not None

    # Attach electronic structure data
    with pytest.raises(
        ValueError,
        match="Attaching electronic structure files to a molecule vault hosting a 2D molecule is "
        "not allowed.",
    ):
        f.attach_electronic_structure(electronic_structure_data="irrelevant", state="n")
    assert f.mol_vault is not None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.attach_electronic_structure
@pytest.mark.parametrize(
    "electronic_structure_data, state, _preattach, _error_type",
    [
        (False, "n", False, TypeError),
        ("dummy_electronic_struc_file_01.molden", ["n", "n+1"], False, TypeError),
        ("dummy_electronic_struc_file_01.molden", "n-2", False, ValueError),
        (
            ["dummy_electronic_struc_file_01.molden", "dummy_electronic_struc_file_02.molden"],
            "n",
            False,
            ValueError,
        ),
        ("i_dont_exist.molden", "n", False, FileNotFoundError),
        ("dummy_electronic_struc_file_02.molden", "n", True, ValueError),
    ],
)
def test_attach_electronic_structure4(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
    electronic_structure_data: Any,
    state: Any,
    _preattach: bool,
    _error_type: Any,
) -> None:
    """Test for the ``attach_electronic_structure()`` method: invalid inputs."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    _file_path = fetch_data_file(file_name="clopidogrel-conf_01.xyz")
    f.read_input(input_value=_file_path, namespace="irrelevant", input_format="file")
    assert f.mol_vault is not None

    # Preattach electronic structure data if requested
    if _preattach is True:
        _file_path2 = fetch_data_file(file_name="dummy_electronic_struc_file_01.molden")
        f.attach_electronic_structure(electronic_structure_data=_file_path2, state="n")

    # Attach electronic structure data
    with pytest.raises(_error_type):
        f.attach_electronic_structure(
            electronic_structure_data=electronic_structure_data, state=state
        )

    if _preattach is False:
        assert f.mol_vault.electronic_strucs_n == []
        assert f.mol_vault.electronic_strucs_n_minus1 == []
        assert f.mol_vault.electronic_strucs_n_plus1 == []

        assert f.mol_vault.electronic_struc_types_n == []
        assert f.mol_vault.electronic_struc_types_n_minus1 == []
        assert f.mol_vault.electronic_struc_types_n_plus1 == []

    else:
        assert f.mol_vault.electronic_strucs_n == [_file_path2]
        assert f.mol_vault.electronic_strucs_n_minus1 == []
        assert f.mol_vault.electronic_strucs_n_plus1 == []

        assert f.mol_vault.electronic_struc_types_n == ["molden"]
        assert f.mol_vault.electronic_struc_types_n_minus1 == []
        assert f.mol_vault.electronic_struc_types_n_plus1 == []

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.attach_electronic_structure
def test_attach_electronic_structure5(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    fetch_data_file: Callable[[str], str],
) -> None:
    """Test for the ``attach_electronic_structure()`` method: attach None."""
    # Setup featurizer and read input
    f = fresh_featurizer()
    assert f.mol_vault is None

    _file_path = fetch_data_file(file_name="radical_cation_e.sdf")
    f.read_input(input_value=_file_path, namespace="irrelevant", input_format="file")
    assert f.mol_vault is not None

    # Attach electronic structure data
    _path = fetch_data_file(file_name="dummy_electronic_struc_file_01.molden")
    electronic_structure_data = [_path, None]
    f.attach_electronic_structure(electronic_structure_data=electronic_structure_data, state="n+1")

    assert f.mol_vault.electronic_strucs_n == []
    assert f.mol_vault.electronic_strucs_n_minus1 == []
    assert f.mol_vault.electronic_strucs_n_plus1 == [_path, None]

    assert f.mol_vault.electronic_struc_types_n == []
    assert f.mol_vault.electronic_struc_types_n_minus1 == []
    assert f.mol_vault.electronic_struc_types_n_plus1 == ["molden", None]

    # Check logs
    assert len(caplog.records) > 0

    _log_found = False
    _target_str = (
        "Electronic structure data of type None was attached to conformer with index 1 "
        "for state 'n+1'."
    )
    for record in caplog.records:
        if record.levelno == logging.WARNING and _target_str in record.msg:
            _log_found = True
            break
    assert _log_found is True

    # Clean up
    clean_up_logfile()
