"""Test functions for the ``bonafide.utils.base_featurizer`` module."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, List

import pytest
from pytest_mock import MockerFixture
from rdkit import Chem

from bonafide.utils.base_featurizer import BaseFeaturizer

#########################################################
# Helper classes and functions for the individual tests #
#########################################################


class FeatureFactoryValid(BaseFeaturizer):
    """Valid dummy feature factory with mocked calculated feature."""

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        self.results[self.atom_bond_idx] = {self.feature_name: self.dummy_output}


class FeatureFactoryBrokenCalculate(BaseFeaturizer):
    """Invalid dummy feature factory with an error in the calculate method."""

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        res = 42 / 0
        self.results[self.atom_bond_idx] = {self.feature_name: self.dummy_output}


def _raiser_function() -> None:
    raise ValueError("Naaa ... not today.")


class FeatureFactoryRaiseError(BaseFeaturizer):
    """In general valid dummy feature factory with a calculate method that raises an error."""

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        _raiser_function()
        self.results[self.atom_bond_idx] = {self.feature_name: self.dummy_output}


class FeatureFactorySetError(BaseFeaturizer):
    """In general valid dummy feature factory with a calculate method that sets an error."""

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        self._err = "Today, I'm not gonna do anything."
        self.results[self.atom_bond_idx] = {self.feature_name: self.dummy_output}


class FeatureFactoryNoCalculate(BaseFeaturizer):
    """Invalid dummy feature factory with missing calculate method."""

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()


class FeatureFactoryNoExtraction(BaseFeaturizer):
    """Invalid dummy feature factory with missing extraction_mode attribute."""

    def __init__(self) -> None:
        super().__init__()

    def calculate(self) -> None:
        pass


class FeatureFactoryWrongExtraction(BaseFeaturizer):
    """Invalid dummy feature factory with missing extraction_mode attribute."""

    def __init__(self) -> None:
        self.conformer_name = "test"
        self.extraction_mode = "parma ham is best"
        super().__init__()

    def calculate(self) -> None:
        pass


####################################
# Tests for the __init__() method. #
####################################


@pytest.mark.base_featurizer_init
def test___init__() -> None:
    """Test for the ``__init__()`` method."""
    f = FeatureFactoryValid()
    assert hasattr(f, "results") is True
    assert hasattr(f, "_out") is True
    assert hasattr(f, "_err") is True
    assert hasattr(f, "extraction_mode") is True
    assert hasattr(f, "calculate") is True


####################################
# Tests for the __call__() method. #
####################################


@pytest.mark.base_featurizer_call
def test___call__(caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``__call__()`` method: valid feature calculation."""
    # Dummy parameters for testing
    _conformer_name = "irrelevant"
    _conformer_idx = 3
    _atom_bond_idx = 1
    _feature_value = 42
    _feature_name = "test_feature_name"
    _cache = [{}] * 10

    # Run the feature calculation
    run_feature_calc = FeatureFactoryValid()
    output, error_message = run_feature_calc(
        conformer_name=_conformer_name,
        feature_name=_feature_name,
        conformer_idx=_conformer_idx,
        atom_bond_idx=_atom_bond_idx,
        feature_cache=_cache,
        _keep_output_files=True,
        dummy_output=_feature_value,
    )

    # Check featurizer object
    assert run_feature_calc.conformer_name == _conformer_name
    assert run_feature_calc.feature_name == _feature_name
    assert run_feature_calc.conformer_idx == _conformer_idx
    assert run_feature_calc.atom_bond_idx == _atom_bond_idx
    assert run_feature_calc._keep_output_files is True
    assert run_feature_calc.dummy_output == _feature_value
    assert run_feature_calc.feature_cache[_conformer_idx] == {
        _feature_name: {_atom_bond_idx: _feature_value}
    }

    # Check output
    assert output == _feature_value
    assert error_message is None

    # Check logging
    assert len(caplog.records) == 0


@pytest.mark.base_featurizer_call
@pytest.mark.parametrize(
    "feature_factor", [FeatureFactoryBrokenCalculate, FeatureFactoryRaiseError]
)
def test___call__2(caplog: pytest.LogCaptureFixture, feature_factor: type) -> None:
    """Test for the ``__call__()`` method: broken feature calculation."""
    # Dummy parameters for testing
    _conformer_name = "irrelevant"
    _conformer_idx = 3
    _atom_bond_idx = 1
    _feature_value = 42
    _feature_name = "test_feature_name"
    _feature_type = "bond"
    _cache = [{}] * 10

    # Run the feature calculation
    run_feature_calc = feature_factor()

    with pytest.raises(
        RuntimeError, match="An unexpected error occurred during the calculation of the"
    ):
        run_feature_calc(
            conformer_name=_conformer_name,
            feature_name=_feature_name,
            feature_type=_feature_type,
            conformer_idx=_conformer_idx,
            atom_bond_idx=_atom_bond_idx,
            feature_cache=_cache,
            _keep_output_files=True,
            dummy_output=_feature_value,
        )

    # Check output
    assert run_feature_calc.feature_cache == _cache

    # Check logging
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    os.chdir("..")
    _work_dir_name = f"_w__{_conformer_name}__"
    _work_dirs = [
        x for x in os.listdir() if x.startswith(_work_dir_name) is True and os.path.isdir(x) is True
    ]
    for dir in _work_dirs:
        shutil.rmtree(dir)


@pytest.mark.base_featurizer_call
def test___call__3(caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``__call__()`` method: feature calculation sets error."""
    # Dummy parameters for testing
    _conformer_name = "irrelevant"
    _conformer_idx = 3
    _atom_bond_idx = 1
    _feature_value = 42
    _feature_name = "test_feature_name"
    _cache = [{}] * 10

    # Run the feature calculation
    run_feature_calc = FeatureFactorySetError()
    output, error_message = run_feature_calc(
        conformer_name=_conformer_name,
        feature_name=_feature_name,
        conformer_idx=_conformer_idx,
        atom_bond_idx=_atom_bond_idx,
        feature_cache=_cache,
        _keep_output_files=True,
        dummy_output=_feature_value,
    )

    # Check output
    assert output is None
    assert error_message is not None
    assert type(error_message) == str
    assert run_feature_calc.feature_cache == _cache

    # Check logging
    assert len(caplog.records) == 0


###############################################
# Tests for the _check_requirements() method. #
###############################################


@pytest.mark.base_featurizer_check_requirements
def test__check_requirements(caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``_check_requirements()`` method: missing calculate method."""
    with pytest.raises(
        NotImplementedError, match=r"calculate\(\) method must be implemented in child class "
    ):
        run_feature_calc = FeatureFactoryNoCalculate()
        run_feature_calc(conformer_name="irrelevant")

    # Check logging
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)


@pytest.mark.base_featurizer_check_requirements
def test__check_requirements2(caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``_check_requirements()`` method: missing extraction_mode attribute."""
    with pytest.raises(
        AttributeError, match="Attribute 'extraction_mode' must be set in child class"
    ):
        run_feature_calc = FeatureFactoryNoExtraction()
        run_feature_calc(conformer_name="irrelevant")

    # Check logging
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)


@pytest.mark.base_featurizer_check_requirements
def test__check_requirements3(caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``_check_requirements()`` method: wrong extraction_mode attribute."""
    with pytest.raises(
        ValueError, match="'extraction_mode' must be set either to 'single' or 'multi'"
    ):
        run_feature_calc = FeatureFactoryWrongExtraction()
        run_feature_calc(conformer_name="irrelevant")

    # Check logging
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)


#######################################
# Tests for the _from_cache() method. #
#######################################


@pytest.mark.base_featurizer_from_cache
@pytest.mark.parametrize("successful", [True, False])
def test__from_cache(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture, successful: bool
) -> None:
    """Test for the ``_from_cache()`` method."""
    # Dummy parameters for testing
    _conformer_idx = 3
    _atom_bond_idx = 1
    _feature_value = "42.0"
    _feature_name = "test_feature_name"
    _cache = [{}] * 10

    # Patch requirement checking
    mock = mocker.patch("bonafide.utils.base_featurizer.BaseFeaturizer._check_requirements")

    # Set up BaseFeaturizer object with cached value
    bf = BaseFeaturizer()
    bf.feature_name = _feature_name
    bf.conformer_idx = _conformer_idx
    bf.atom_bond_idx = _atom_bond_idx
    bf.feature_cache = _cache
    bf.feature_cache[_conformer_idx] = {_feature_name: {_atom_bond_idx: _feature_value}}

    if successful is False:
        bf.feature_name = "i_was_not_calculated_yet"

    # Get value from cache
    assert bf._out is None
    bf._from_cache()

    if successful is True:
        assert bf._out == _feature_value
    else:
        assert bf._out is None

    assert mock.call_count == 1

    # Check logging
    assert len(caplog.records) == 0


#####################################
# Tests for the _to_cache() method. #
#####################################

SMILES_READING_PARAMS = Chem.SmilesParserParams()
SMILES_READING_PARAMS.removeHs = False
SMILES_READING_PARAMS.sanitize = True


@pytest.mark.base_featurizer_to_cache
@pytest.mark.parametrize(
    "feature_type, extraction_mode, mocked_feature_factory_output, expected_feature_cache",
    [
        ("atom", "single", {3: {"test_feature_name": True}}, [{"test_feature_name": {3: True}}]),
        (
            "atom",
            "single",
            {
                3: {"test_feature_name": True, "test_feature_name2": False},
                4: {"test_feature_name": 13},
                5: {"test_feature_name2": "completed"},
            },
            [
                {
                    "test_feature_name": {3: True, 4: 13},
                    "test_feature_name2": {3: False, 5: "completed"},
                }
            ],
        ),
        (
            "atom",
            "multi",
            {0: {"test_feature_name": 123.123}, 3: {"test_feature_name": True}},
            [
                {
                    "test_feature_name": {
                        0: 123.123,
                        1: "_inaccessible",
                        2: "_inaccessible",
                        3: True,
                        4: "_inaccessible",
                        5: "_inaccessible",
                    }
                }
            ],
        ),
        (
            "atom",
            "multi",
            {
                0: {"test_feature_name": 123.123, "test_feature_name2": 10},
                1: {"test_feature_name3": 15},
                3: {"test_feature_name": True},
            },
            [
                {
                    "test_feature_name": {
                        0: 123.123,
                        1: "_inaccessible",
                        2: "_inaccessible",
                        3: True,
                        4: "_inaccessible",
                        5: "_inaccessible",
                    },
                    "test_feature_name2": {
                        0: 10,
                        1: "_inaccessible",
                        2: "_inaccessible",
                        3: "_inaccessible",
                        4: "_inaccessible",
                        5: "_inaccessible",
                    },
                    "test_feature_name3": {
                        0: "_inaccessible",
                        1: 15,
                        2: "_inaccessible",
                        3: "_inaccessible",
                        4: "_inaccessible",
                        5: "_inaccessible",
                    },
                }
            ],
        ),
        (
            "atom",
            "single",
            {
                0: {"test_feature_name": 123.123, "test_feature_name2": 10},
                1: {"test_feature_name3": 15},
                3: {"test_feature_name": True},
            },
            [
                {
                    "test_feature_name": {0: 123.123, 3: True},
                    "test_feature_name2": {0: 10},
                    "test_feature_name3": {1: 15},
                }
            ],
        ),
        (
            "bond",
            "multi",
            {0: {"test_feature_name": 123.123}, 3: {"test_feature_name": True}},
            [
                {
                    "test_feature_name": {
                        0: 123.123,
                        1: "_inaccessible",
                        2: "_inaccessible",
                        3: True,
                        4: "_inaccessible",
                    }
                }
            ],
        ),
        (
            "bond",
            "multi",
            {
                0: {"test_feature_name": 123.123, "test_feature_name2": 10},
                3: {"test_feature_name": True},
            },
            [
                {
                    "test_feature_name": {
                        0: 123.123,
                        1: "_inaccessible",
                        2: "_inaccessible",
                        3: True,
                        4: "_inaccessible",
                    },
                    "test_feature_name2": {
                        0: 10,
                        1: "_inaccessible",
                        2: "_inaccessible",
                        3: "_inaccessible",
                        4: "_inaccessible",
                    },
                }
            ],
        ),
    ],
)
def test__to_cache(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    feature_type: str,
    extraction_mode: str,
    mocked_feature_factory_output: Dict[int, Dict[str, Any]],
    expected_feature_cache: List[Dict[int, Dict[str, Any]]],
) -> None:
    """Test for the ``_to_cache()`` method."""
    # Dummy parameters for testing
    _cache = [{}]
    _mol = Chem.MolFromSmiles("[H]OC([H])([H])F", params=SMILES_READING_PARAMS)

    # Patch requirement checking
    mock = mocker.patch("bonafide.utils.base_featurizer.BaseFeaturizer._check_requirements")

    # Set up BaseFeaturizer object
    bf = BaseFeaturizer()
    bf.feature_type = feature_type
    bf.extraction_mode = extraction_mode
    bf.mol = _mol
    bf.conformer_idx = 0
    bf.feature_cache = _cache
    bf.results = {key: value for key, value in mocked_feature_factory_output.items()}

    # Write results to cache
    bf._to_cache()

    # Check feature cache
    assert bf.feature_cache == expected_feature_cache

    assert mock.call_count == 1

    # Check logging
    assert len(caplog.records) == 0


@pytest.mark.base_featurizer_to_cache
def test__to_cache2(mocker: MockerFixture, caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``_to_cache()`` method: empty results dictionary."""
    _cache = [{}] * 10

    # Patch requirement checking
    mock = mocker.patch("bonafide.utils.base_featurizer.BaseFeaturizer._check_requirements")

    bf = BaseFeaturizer()
    bf.feature_cache = _cache
    bf._to_cache()
    assert bf.feature_cache == _cache

    assert mock.call_count == 1

    # Check logging
    assert len(caplog.records) == 0
