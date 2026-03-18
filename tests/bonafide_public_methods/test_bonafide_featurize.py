"""Test functions for the ``featurize_atoms()`` and ``featurize_bonds()`` methods in the
``bonafide.bonafide`` module.
"""

from __future__ import annotations

import logging
import shutil
import warnings
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np
import pytest
from pytest_mock import MockerFixture
from rdkit import RDLogger

if TYPE_CHECKING:
    from pytest import MonkeyPatch

    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


#####################################################################
# Helper classes and data for testing of the featurization methods. #
#####################################################################


class _TestingBaseFeaturizer:
    def __init__(self) -> None:
        super().__init__()
        self.results = {}

    def __call__(self, **kwargs):
        for attr_name, value in kwargs.items():
            setattr(self, attr_name, value)
        self.calculate()

        return self.results[self.atom_bond_idx][self.feature_name], None

    def calculate(self) -> None:
        self.results[self.atom_bond_idx] = {self.feature_name: self.testing_option}


# Atom classes
class Testing2DAtomStrFeature(_TestingBaseFeaturizer):
    def __init__(self) -> None:
        super().__init__()


class Testing2DAtomBoolFeature(_TestingBaseFeaturizer):
    def __init__(self) -> None:
        super().__init__()


class Testing2DAtomIntFeature(_TestingBaseFeaturizer):
    def __init__(self) -> None:
        super().__init__()


class Testing2DAtomFloatFeature(_TestingBaseFeaturizer):
    def __init__(self) -> None:
        super().__init__()


# Bond classes
class Testing2DBondStrFeature(_TestingBaseFeaturizer):
    def __init__(self) -> None:
        super().__init__()


class Testing2DBondBoolFeature(_TestingBaseFeaturizer):
    def __init__(self) -> None:
        super().__init__()


class Testing2DBondIntFeature(_TestingBaseFeaturizer):
    def __init__(self) -> None:
        super().__init__()


class Testing2DBondFloatFeature(_TestingBaseFeaturizer):
    def __init__(self) -> None:
        super().__init__()


# Classes for invalid/inaccessible cases
class Testing2DAtomFeatureError:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, **kwargs):
        for attr_name, value in kwargs.items():
            setattr(self, attr_name, value)
        return None, "Intentional error for testing purposes"


class Testing2DAtomFeatureNone:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, **kwargs):
        for attr_name, value in kwargs.items():
            setattr(self, attr_name, value)
        return None, None


class Testing2DAtomFeatureInaccessible:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, **kwargs):
        for attr_name, value in kwargs.items():
            setattr(self, attr_name, value)
        return "_inaccessible", None


class Testing2DAtomFeatureWrongDatatype:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, **kwargs):
        for attr_name, value in kwargs.items():
            setattr(self, attr_name, value)
        return np.array([1, 2, 3]), None


TEST_FEATURE_INFO = {
    0: {
        "name": "testing2D-atom-str_feature",
        "origin": "testing",
        "feature_type": "atom",
        "dimensionality": "2D",
        "data_type": "str",
        "requires_electronic_structure_data": False,
        "requires_bond_data": False,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DAtomStrFeature",
    },
    1: {
        "name": "testing2D-atom-bool_feature",
        "origin": "testing",
        "feature_type": "atom",
        "dimensionality": "2D",
        "data_type": "bool",
        "requires_electronic_structure_data": False,
        "requires_bond_data": False,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DAtomBoolFeature",
    },
    2: {
        "name": "testing2D-atom-int_feature",
        "origin": "testing",
        "feature_type": "atom",
        "dimensionality": "2D",
        "data_type": "int",
        "requires_electronic_structure_data": False,
        "requires_bond_data": False,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DAtomIntFeature",
    },
    3: {
        "name": "testing2D-atom-float_feature",
        "origin": "testing",
        "feature_type": "atom",
        "dimensionality": "2D",
        "data_type": "float",
        "requires_electronic_structure_data": False,
        "requires_bond_data": False,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DAtomFloatFeature",
    },
    4: {
        "name": "testing2D-bond-str_feature",
        "origin": "testing",
        "feature_type": "bond",
        "dimensionality": "2D",
        "data_type": "str",
        "requires_electronic_structure_data": False,
        "requires_bond_data": True,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DBondStrFeature",
    },
    5: {
        "name": "testing2D-bond-bool_feature",
        "origin": "testing",
        "feature_type": "bond",
        "dimensionality": "2D",
        "data_type": "bool",
        "requires_electronic_structure_data": False,
        "requires_bond_data": True,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DBondBoolFeature",
    },
    6: {
        "name": "testing2D-bond-int_feature",
        "origin": "testing",
        "feature_type": "bond",
        "dimensionality": "2D",
        "data_type": "int",
        "requires_electronic_structure_data": False,
        "requires_bond_data": True,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DBondIntFeature",
    },
    7: {
        "name": "testing2D-bond-float_feature",
        "origin": "testing",
        "feature_type": "bond",
        "dimensionality": "2D",
        "data_type": "float",
        "requires_electronic_structure_data": False,
        "requires_bond_data": True,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DBondFloatFeature",
    },
    8: {
        "name": "testing2D-atom-feature_error",
        "origin": "testing",
        "feature_type": "atom",
        "dimensionality": "2D",
        "data_type": "str",
        "requires_electronic_structure_data": False,
        "requires_bond_data": False,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DAtomFeatureError",
    },
    9: {
        "name": "testing2D-atom-feature_none",
        "origin": "testing",
        "feature_type": "atom",
        "dimensionality": "2D",
        "data_type": "str",
        "requires_electronic_structure_data": False,
        "requires_bond_data": False,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DAtomFeatureNone",
    },
    10: {
        "name": "testing2D-atom-feature_inaccessible",
        "origin": "testing",
        "feature_type": "atom",
        "dimensionality": "2D",
        "data_type": "str",
        "requires_electronic_structure_data": False,
        "requires_bond_data": False,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DAtomFeatureInaccessible",
    },
    11: {
        "name": "testing2D-atom-feature_wrong_datatype",
        "origin": "testing",
        "feature_type": "atom",
        "dimensionality": "2D",
        "data_type": "str",
        "requires_electronic_structure_data": False,
        "requires_bond_data": False,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DAtomFeatureWrongDatatype",
    },
    12: {
        "name": "testing2D-atom-feature_wrong_datatype",
        "origin": "testing",
        "feature_type": "atom",
        "dimensionality": "2D",
        "data_type": "ndarray",
        "requires_electronic_structure_data": False,
        "requires_bond_data": False,
        "requires_charge": False,
        "requires_multiplicity": False,
        "config_path": "testing.config.path",
        "factory": "Testing2DAtomFeatureWrongDatatype",
    },
}


TEST_FEATURE_FACTORIES = {
    "Testing2DAtomStrFeature": Testing2DAtomStrFeature,
    "Testing2DAtomBoolFeature": Testing2DAtomBoolFeature,
    "Testing2DAtomIntFeature": Testing2DAtomIntFeature,
    "Testing2DAtomFloatFeature": Testing2DAtomFloatFeature,
    "Testing2DBondStrFeature": Testing2DBondStrFeature,
    "Testing2DBondBoolFeature": Testing2DBondBoolFeature,
    "Testing2DBondIntFeature": Testing2DBondIntFeature,
    "Testing2DBondFloatFeature": Testing2DBondFloatFeature,
    "Testing2DAtomFeatureError": Testing2DAtomFeatureError,
    "Testing2DAtomFeatureNone": Testing2DAtomFeatureNone,
    "Testing2DAtomFeatureInaccessible": Testing2DAtomFeatureInaccessible,
    "Testing2DAtomFeatureWrongDatatype": Testing2DAtomFeatureWrongDatatype,
}


#########################################################################
# General tests for the featurize_atoms() and featurize_bonds() method. #
#########################################################################


@pytest.mark.featurize_atoms_featurize_bonds
@pytest.mark.parametrize(
    "input_string, input_format, feature_idx, dummy_result, _feature_type",
    [
        # Valid cases
        ("O(C([H])([H])F)[H]", "smiles", 0, "correct_str_value", "atom"),
        ("O(C([H])([H])F)[H]", "smiles", 1, True, "atom"),
        ("O(C([H])([H])F)[H]", "smiles", 2, 42, "atom"),
        ("O(C([H])([H])F)[H]", "smiles", 3, 123.123, "atom"),
        ("O(C([H])([H])F)[H]", "smiles", 4, "also_a_correct_str_value", "bond"),
        ("O(C([H])([H])F)[H]", "smiles", 5, False, "bond"),
        ("O(C([H])([H])F)[H]", "smiles", 6, 420, "bond"),
        ("O(C([H])([H])F)[H]", "smiles", 7, 1230.123, "bond"),
        ("spiro_mol-conf_00.sdf", "file", 0, "correct_str_value", "atom"),
        ("spiro_mol-conf_00.sdf", "file", 1, True, "atom"),
        ("spiro_mol-conf_00.sdf", "file", 2, 42, "atom"),
        ("spiro_mol-conf_00.sdf", "file", 3, 123.123, "atom"),
        ("spiro_mol-conf_00.sdf", "file", 4, "also_a_correct_str_value", "bond"),
        ("spiro_mol-conf_00.sdf", "file", 5, False, "bond"),
        ("spiro_mol-conf_00.sdf", "file", 6, 420, "bond"),
        ("spiro_mol-conf_00.sdf", "file", 7, 1230.123, "bond"),
        # Invalid cases (errors or warnings), separate testing for bonds not required
        ("O(C([H])([H])F)[H]", "smiles", 8, "irrelevant", "atom"),
        ("O(C([H])([H])F)[H]", "smiles", 9, "irrelevant", "atom"),
        ("O(C([H])([H])F)[H]", "smiles", 10, "irrelevant", "atom"),
        ("O(C([H])([H])F)[H]", "smiles", 11, "irrelevant", "atom"),
        ("O(C([H])([H])F)[H]", "smiles", 12, "irrelevant", "atom"),
        ("clopidogrel-conf_01.xyz", "file", 8, "irrelevant", "atom"),
        ("clopidogrel-conf_01.xyz", "file", 9, "irrelevant", "atom"),
        ("clopidogrel-conf_01.xyz", "file", 10, "irrelevant", "atom"),
        ("clopidogrel-conf_01.xyz", "file", 11, "irrelevant", "atom"),
        ("clopidogrel-conf_01.xyz", "file", 12, "irrelevant", "atom"),
    ],
)
def test_featurize_atoms_featurize_bonds(
    monkeypatch: MonkeyPatch,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    fetch_data_file: Callable[[str], str],
    clean_up_logfile: Callable[[str], None],
    input_string: str,
    input_format: str,
    feature_idx: int,
    dummy_result: Union[str, bool, int, float],
    _feature_type: str,
) -> None:
    """Test for the ``featurize_atoms()`` and ``featurize_bonds()`` method: valid and invalid
    examples.
    """
    # Explicit import to allow monkeypatching
    from bonafide import _bonafide

    # Setup featurizer
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Read input
    _dimensionality = "2D"
    if input_format.lower().strip() == "file":
        input_string = fetch_data_file(file_name=input_string)
        _dimensionality = "3D"

    f.read_input(
        input_value=input_string,
        namespace="irrelevant",
        input_format=input_format,
        output_directory="irrelevant_out_dir",
    )
    assert f.mol_vault is not None

    # Modify _feature_info for testing
    f._feature_info = TEST_FEATURE_INFO

    # Mock the checking of the feature indices
    mock_check_feature_indices = mocker.patch(
        "bonafide._bonafide_utils._AtomBondFeaturizerUtils._check_feature_indices",
        return_value=[feature_idx],
    )

    # Mock the extraction of the configuration settings
    mock_get_configs = mocker.patch(
        "bonafide._bonafide_utils._AtomBondFeaturizerUtils._get_configs",
        return_value={"irrelevant_config_key": True},
    )

    # Mock the configuration settings validation
    mock_config_data_validator = mocker.patch(
        "bonafide._bonafide.config_data_validator",
        return_value={"testing_option": dummy_result},
    )

    # Monkeypatch the FEATURE_FACTORIES dictionary
    monkeypatch.setattr(_bonafide, "FEATURE_FACTORIES", TEST_FEATURE_FACTORIES)

    # Featurize atoms or bonds
    _idx = 2
    if _feature_type == "atom" and feature_idx < 11:
        f.featurize_atoms(atom_indices=_idx, feature_indices=feature_idx)
    elif _feature_type == "bond" and feature_idx < 11:
        f.featurize_bonds(bond_indices=_idx, feature_indices=feature_idx)
    elif _feature_type == "atom" and feature_idx == 11:
        with pytest.raises(RuntimeError, match="does not match the expected data type 'str'."):
            f.featurize_atoms(atom_indices=_idx, feature_indices=feature_idx)
    elif _feature_type == "atom" and feature_idx == 12:
        with pytest.raises(
            RuntimeError,
            match="is not supported. Supported feature types are int, float, bool, and str.",
        ):
            f.featurize_atoms(atom_indices=_idx, feature_indices=feature_idx)

    # Check mocks
    mock_check_feature_indices.assert_called_once_with(
        feature_indices=feature_idx, feature_type=_feature_type, dimensionality=_dimensionality
    )

    mock_get_configs.assert_called_with(
        key_list=["testing", "config", "path"], include_root_data=True
    )  # is called twice, during _run_featurization and from _rearrange_feature_indices

    mock_config_data_validator.assert_called_once_with(
        config_path=["testing", "config", "path"],
        params={"irrelevant_config_key": True, "feature_info": TEST_FEATURE_INFO},
        _namespace="irrelevant",
    )

    # Check results (successful cases)
    if _feature_type == "atom" and feature_idx < 8:
        assert (
            f.mol_vault.mol_objects[0]
            .GetAtomWithIdx(_idx)
            .GetPropsAsDict()[TEST_FEATURE_INFO[feature_idx]["name"]]
            == dummy_result
        )
    elif _feature_type == "bond" and feature_idx < 8:
        assert (
            f.mol_vault.mol_objects[0]
            .GetBondWithIdx(_idx)
            .GetPropsAsDict()[TEST_FEATURE_INFO[feature_idx]["name"]]
            == dummy_result
        )

    # Check results (unsuccessful cases)
    elif _feature_type == "atom" and feature_idx >= 8 and feature_idx != 10:
        assert f.mol_vault.mol_objects[0].GetAtomWithIdx(_idx).GetPropsAsDict() == {}
    elif _feature_type == "bond" and feature_idx >= 8 and feature_idx != 10:
        assert f.mol_vault.mol_objects[0].GetBondWithIdx(_idx).GetPropsAsDict() == {}

    # Check results (inaccessible case)
    elif _feature_type == "atom" and feature_idx == 10:
        assert (
            f.mol_vault.mol_objects[0]
            .GetAtomWithIdx(_idx)
            .GetPropsAsDict()[TEST_FEATURE_INFO[feature_idx]["name"]]
            == "_inaccessible"
        )
    elif _feature_type == "bond" and feature_idx == 10:
        assert (
            f.mol_vault.mol_objects[0]
            .GetBondWithIdx(_idx)
            .GetPropsAsDict()[TEST_FEATURE_INFO[feature_idx]["name"]]
            == "_inaccessible"
        )

    # Check logs
    assert len(caplog.records) > 0

    if feature_idx < 8:
        assert all(record.levelno == logging.INFO for record in caplog.records)
    elif feature_idx == 8:
        assert any(
            "Intentional error for testing purposes" in record.msg for record in caplog.records
        )
    elif feature_idx == 9:
        assert any(
            "terminated without error but the feature value is None for conformer with index"
            in record.msg
            for record in caplog.records
        )
    elif feature_idx == 10:
        assert any(
            "terminated without error but the feature value is '_inaccessible'" in record.msg
            for record in caplog.records
        )
    elif feature_idx == 11:
        assert any(
            "does not match the expected data type 'str'." in record.msg
            for record in caplog.records
        )
    elif feature_idx == 12:
        assert any(
            "is not supported. Supported feature types are int, float, bool, and str." in record.msg
            for record in caplog.records
        )

    # Clean up
    shutil.rmtree(path="irrelevant_out_dir")
    clean_up_logfile()


###########################################
# Tests for the featurize_atoms() method. #
###########################################


@pytest.mark.featurize_atoms
def test_featurize_atoms(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``featurize_atoms()`` method: fails because no molecule read yet."""
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Featurize atoms
    with pytest.raises(ValueError, match="featurizing atoms"):
        f.featurize_atoms(atom_indices=[1, 2, 3], feature_indices=[42, 142])
    assert f.mol_vault is None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.featurize_atoms
# Attention: specific feature index used, might change and will result in breaking the test
# Feature should be "mendeleev2D-atom-en_pauling" (index 134 in the current implementation)
@pytest.mark.parametrize(
    "atom_indices, feature_indices, error_string, _error_type",
    [
        ("1, 2, 3", 134, "Invalid input to 'atom_indices': ", ValueError),
        ([1, 2, "3"], 134, "Invalid input to 'atom_indices': ", TypeError),
        (True, 134, "Invalid input to 'atom_indices': ", TypeError),
        (-10, 134, "Invalid input to 'atom_indices': ", ValueError),
        ([1, 2, -10], 134, "Invalid input to 'atom_indices': ", ValueError),
        ([1, 2, 1000000], 134, "Invalid input to 'atom_indices': ", ValueError),
        ([0, 1, 2], "[1, 2, 3]", "Invalid input to 'feature_indices': ", ValueError),
        ([0, 1, 2], None, "Invalid input to 'feature_indices': ", TypeError),
        ([0, 1, 2], [True, False], "Invalid input to 'feature_indices': ", TypeError),
        ([0, 1, 2], [0, 1], "Invalid input to 'feature_indices': ", ValueError),
    ],
)
def test_featurize_atoms2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    atom_indices: Any,
    feature_indices: Any,
    error_string: str,
    _error_type: Any,
) -> None:
    """Test for the ``featurize_atoms()`` method: invalid inputs."""
    # Setup featurizer
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Read input
    f.read_input(input_value="O(C([H])([H])F)[H]", namespace="irrelevant", input_format="smiles")
    assert f.mol_vault is not None

    # Featurize atoms
    with pytest.raises(_error_type, match=error_string):
        f.featurize_atoms(atom_indices=atom_indices, feature_indices=feature_indices)

    for mol in f.mol_vault.mol_objects:
        for atom in mol.GetAtoms():
            assert atom.GetPropsAsDict() == {}

    assert f.mol_vault.atom_feature_cache_n == [{} for _ in f.mol_vault.mol_objects]

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.featurize_atoms
@pytest.mark.parametrize(
    "case_id, _error_type",
    [
        ("el_struc_error", ValueError),
        ("bond_error", ValueError),
        ("charge_error", ValueError),
        ("multiplicity_error", ValueError),
    ],
)
def test_featurize_atoms3(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    fetch_data_file: Callable[[str], str],
    clean_up_logfile: Callable[[str], None],
    case_id: str,
    _error_type: Any,
) -> None:
    """Test for the ``featurize_atoms()`` method: inappropriate mol vaults."""
    # Setup featurizer
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Read input
    input_string = fetch_data_file(file_name="spiro_mol_e.xyz")
    f.read_input(input_value=input_string, namespace="irrelevant", input_format="file")
    assert f.mol_vault is not None

    # Helper DataFrame with feature info
    _df = f.list_atom_features()

    # Featurize atoms for different cases
    if case_id == "el_struc_error":
        fidx = _df[_df["requires_electronic_structure_data"]].index.tolist()[0]
        with pytest.raises(
            _error_type, match="electronic structure data is required but is not available."
        ):
            f.featurize_atoms(atom_indices=0, feature_indices=[fidx])

    if case_id == "bond_error":
        fidx = _df[_df["requires_bond_data"]].index.tolist()[-1]
        with pytest.raises(_error_type, match="bond data is required but is not available."):
            f.featurize_atoms(atom_indices=0, feature_indices=[fidx])

    if case_id == "charge_error":
        fidx = _df[_df["requires_charge"]].index.tolist()[0]
        with pytest.raises(
            _error_type, match="the charge of the molecule is required but is not set."
        ):
            f.featurize_atoms(atom_indices=0, feature_indices=[fidx])

    if case_id == "multiplicity_error":
        f.set_charge(charge=0)
        fidx = _df[
            _df["requires_multiplicity"] & (_df["requires_electronic_structure_data"] == False)
        ].index.tolist()[0]
        with pytest.raises(
            _error_type, match="the multiplicity of the molecule is required but is not set."
        ):
            f.featurize_atoms(atom_indices=0, feature_indices=[fidx])

    # Assert that no features were added
    for mol in f.mol_vault.mol_objects:
        for atom in mol.GetAtoms():
            assert atom.GetPropsAsDict() == {}

    assert f.mol_vault.atom_feature_cache_n == [{} for _ in f.mol_vault.mol_objects]

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


###########################################
# Tests for the featurize_bonds() method. #
###########################################


@pytest.mark.featurize_bonds
def test_featurize_bonds(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``featurize_bonds()`` method: fails because no molecule read yet."""
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Featurize bonds
    with pytest.raises(ValueError, match="featurizing bonds"):
        f.featurize_bonds(bond_indices=[1, 2, 3], feature_indices=[42, 142])
    assert f.mol_vault is None

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.featurize_bonds
@pytest.mark.parametrize(
    "bond_indices, feature_indices, error_string, _error_type",
    [
        ("1, 2, 3", 0, "Invalid input to 'bond_indices': ", ValueError),
        ([1, 2, "3"], 0, "Invalid input to 'bond_indices': ", TypeError),
        (True, 0, "Invalid input to 'bond_indices': ", TypeError),
        (-10, 0, "Invalid input to 'bond_indices': ", ValueError),
        ([1, 2, -10], 0, "Invalid input to 'bond_indices': ", ValueError),
        ([1, 2, 1000000], 0, "Invalid input to 'bond_indices': ", ValueError),
        ([0, 1, 2], "[1, 2, 3]", "Invalid input to 'feature_indices': ", ValueError),
        ([0, 1, 2], None, "Invalid input to 'feature_indices': ", TypeError),
        ([0, 1, 2], [True, False], "Invalid input to 'feature_indices': ", TypeError),
        (
            [0, 1, 2],
            [2, 3],
            "Invalid input to 'feature_indices': ",
            ValueError,
        ),  # specific feature index used, might change
    ],
)
def test_featurize_bonds2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    bond_indices: Any,
    feature_indices: Any,
    error_string: str,
    _error_type: Any,
) -> None:
    """Test for the ``featurize_bonds()`` method: invalid inputs."""
    # Setup featurizer
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Read input
    f.read_input(input_value="O(C([H])([H])F)[H]", namespace="irrelevant", input_format="smiles")
    assert f.mol_vault is not None

    # Featurize bonds
    with pytest.raises(_error_type, match=error_string):
        f.featurize_bonds(bond_indices=bond_indices, feature_indices=feature_indices)

    for mol in f.mol_vault.mol_objects:
        for bond in mol.GetBonds():
            assert bond.GetPropsAsDict() == {}

    assert f.mol_vault.bond_feature_cache == [{} for _ in f.mol_vault.mol_objects]

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()


@pytest.mark.featurize_bonds
@pytest.mark.parametrize(
    "case_id, _error_type",
    [("el_struc_error", ValueError), ("bond_error", ValueError)],
)
def test_featurize_bonds3(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    fetch_data_file: Callable[[str], str],
    clean_up_logfile: Callable[[str], None],
    case_id: str,
    _error_type: Any,
) -> None:
    """Test for the ``featurize_bonds()`` method: inappropriate mol vaults."""
    # Setup featurizer
    f = fresh_featurizer()
    assert f.mol_vault is None

    # Read input
    input_string = fetch_data_file(file_name="spiro_mol-conf_00.sdf")
    f.read_input(input_value=input_string, namespace="irrelevant", input_format="file")
    assert f.mol_vault is not None

    # Helper DataFrame with feature info
    _df = f.list_bond_features()

    # Featurize bonds for different cases
    if case_id == "el_struc_error":
        fidx = _df[_df["requires_electronic_structure_data"]].index.tolist()[0]
        with pytest.raises(
            _error_type, match="electronic structure data is required but is not available."
        ):
            f.featurize_bonds(bond_indices=0, feature_indices=[fidx])

    if case_id == "bond_error":
        f.mol_vault.bonds_determined = False
        fidx = _df[_df["requires_bond_data"]].index.tolist()[0]
        with pytest.raises(_error_type, match="bond data is required but is not available."):
            f.featurize_bonds(bond_indices=0, feature_indices=[fidx])

    if case_id == "charge_error":
        raise NotImplementedError("There are no bond features requiring the charge to be set.")
    # Just a placeholder to be consistent with test_featurize_atoms3.

    if case_id == "multiplicity_error":
        raise NotImplementedError(
            "There are no bond features requiring the multiplicity to be set."
        )
    # Just a placeholder to be consistent with test_featurize_atoms3.

    # Assert that no features were added
    for mol in f.mol_vault.mol_objects:
        for bond in mol.GetBonds():
            assert bond.GetPropsAsDict() == {}

    assert f.mol_vault.bond_feature_cache == [{} for _ in f.mol_vault.mol_objects]

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()
