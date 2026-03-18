"""Test functions for the ``bonafide.utils.custom_featurizer_input_validation`` module."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict

import pytest

from bonafide.utils.custom_featurizer_input_validation import custom_featurizer_data_validator

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer


class DummyClass:
    pass


dummy_feature_config = {"dummy_option": 1.5}

dummy_feature_info_dict = {
    "name": "custom2D-atom-dummy_example",
    "origin": "custom",
    "feature_type": "Atom",
    "dimensionality": "2d",
    "data_type": "Float",
    "requires_electronic_structure_data": False,
    "requires_bond_data": False,
    "requires_charge": False,
    "requires_multiplicity": False,
    "config_path": dummy_feature_config,
    "factory": DummyClass,
}


################################################################
# Tests for the custom_featurizer_input_validation() function. #
################################################################


@pytest.mark.custom_featurizer_input_validation
def test_custom_featurizer_input_validation(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
) -> None:
    """Test for the ``custom_featurizer_input_validation()`` function: valid input."""
    f = fresh_featurizer()

    # Validate data
    validated_origin, validated_custom_metadata = custom_featurizer_data_validator(
        custom_metadata=dummy_feature_info_dict,
        feature_info=f._feature_info,
        feature_config=f._feature_config,
        namespace="irrelevant",
        loc="irrelevant",
    )
    assert validated_origin == "custom"
    assert type(validated_custom_metadata) == dict

    # Check logs
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)

    # Clean up
    clean_up_logfile()


def _modify_metadata_dict(
    original_dict: Dict[str, Any], modifications: Dict[str, Any]
) -> Dict[str, Any]:
    """Helper function to modify a copy of the original metadata dictionary."""
    modified_dict = copy.deepcopy(original_dict)
    for key, value in modifications.items():
        if value == "_remove" and key in modified_dict:
            del modified_dict[key]
        else:
            modified_dict[key] = value
    return modified_dict


@pytest.mark.custom_featurizer_input_validation
@pytest.mark.parametrize(
    "custom_metadata, _error_type",
    [
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"data_type": "_remove"}
            ),
            KeyError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"i_should_not_be_here": True}
            ),
            KeyError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"name": None}
            ),
            TypeError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"name": ""}
            ),
            ValueError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"origin": ["custom"]}
            ),
            TypeError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"origin": "RDKit"}
            ),
            ValueError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"feature_type": None}
            ),
            TypeError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"feature_type": "global"}
            ),
            ValueError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"dimensionality": False}
            ),
            TypeError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"dimensionality": "0D"}
            ),
            ValueError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"data_type": ["str"]}
            ),
            TypeError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"data_type": "array"}
            ),
            ValueError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict,
                modifications={"requires_electronic_structure_data": "no"},
            ),
            TypeError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"requires_bond_data": "yes"}
            ),
            TypeError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"requires_charge": None}
            ),
            TypeError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"requires_multiplicity": None}
            ),
            TypeError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict,
                modifications={"config_path": ("dummy_option", 1.5)},
            ),
            TypeError,
        ),
        (
            _modify_metadata_dict(
                original_dict=dummy_feature_info_dict, modifications={"factory": "NotAClass"}
            ),
            TypeError,
        ),
    ],
)
def test_custom_featurizer_input_validation2(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    custom_metadata: Dict[str, Any],
    _error_type: Any,
) -> None:
    """Test for the ``custom_featurizer_input_validation()`` function: invalid inputs."""
    f = fresh_featurizer()

    # Validate data
    with pytest.raises(_error_type):
        custom_featurizer_data_validator(
            custom_metadata=custom_metadata,
            feature_info=f._feature_info,
            feature_config=f._feature_config,
            namespace="irrelevant",
            loc="irrelevant",
        )

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()
