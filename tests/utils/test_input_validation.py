"""Test functions for the ``bonafide.utils.input_validation`` module."""

import copy
import json
import logging
import os
import tomllib
from typing import Any, Dict, List

import pytest

import bonafide
from bonafide.utils.input_validation import config_data_validator


@pytest.fixture
def default_configuration_settings() -> Dict[str, Dict[str, Any]]:
    """Load the default configuration settings."""
    _path = os.path.join(os.path.dirname(bonafide.__file__), "_feature_config.toml")
    with open(_path, "rb") as f:
        config = tomllib.load(f)
    return config


@pytest.fixture
def feature_information() -> Dict[int, Dict[str, Any]]:
    """Load the feature information data."""
    _path = os.path.join(os.path.dirname(bonafide.__file__), "_feature_info.json")
    with open(_path, "r") as f:
        info = json.load(f)
    info = {int(idx): data for idx, data in info.items()}
    return info


@pytest.fixture
def all_config_paths() -> List[List[str]]:
    """The individual sections of the default configuration settings file."""
    _paths = [
        ["alfabet"],
        ["bonafide", "autocorrelation"],
        ["bonafide", "constant"],
        ["bonafide", "distance"],
        ["bonafide", "functional_group"],
        ["bonafide", "misc"],
        ["bonafide", "oxidation_state"],
        ["bonafide", "symmetry"],
        ["dbstep"],
        ["dscribe", "coulomb_matrix"],
        ["dscribe", "acsf"],
        ["dscribe", "soap"],
        ["dscribe", "lmbtr"],
        ["kallisto"],
        ["mendeleev"],
        ["morfeus", "buried_volume"],
        ["morfeus", "cone_and_solid_angle"],
        ["morfeus", "dispersion"],
        ["morfeus", "local_force"],
        ["morfeus", "pyramidalization"],
        ["morfeus", "sasa"],
        ["multiwfn"],
        ["multiwfn", "bond_analysis"],
        ["multiwfn", "cdft"],
        ["multiwfn", "fuzzy"],
        ["multiwfn", "misc"],
        ["multiwfn", "orbital"],
        ["multiwfn", "population"],
        ["multiwfn", "surface"],
        ["multiwfn", "topology"],
        ["psi4"],
        ["qmdesc"],
        ["rdkit", "fingerprint"],
        ["rdkit", "misc"],
        ["xtb"],
    ]
    return _paths


################################################################
# Test file format of the default configuration settings file. #
################################################################


@pytest.mark.default_config_file_format
def test_config_file_format(default_configuration_settings: Dict[str, Dict[str, Any]]) -> None:
    """Test the overall data structure of the configuration settings file."""
    assert isinstance(default_configuration_settings, dict)
    for key, value in default_configuration_settings.items():
        assert type(key) == str
        assert type(value) == dict


############################################
# Test all default configuration settings. #
############################################


@pytest.mark.default_settings
def test_default_settings(
    caplog: pytest.LogCaptureFixture,
    all_config_paths: List[List[str]],
    default_configuration_settings: Dict[str, Dict[str, Any]],
    feature_information: Dict[int, Dict[str, Any]],
) -> None:
    """Test all default configuration settings and the respective default values with the
    respective validation classes.
    """
    # Loop over sections
    for config_path in all_config_paths:
        # Skip config_path == ["multiwfn"] because the root parameters are validated with the other
        # Multiwfn sections. ["multiwfn"] was introduced to have the option to separately validate
        # the root parameters without any other parameters in case the root parameters were changed
        # by the user.
        if config_path == ["multiwfn"]:
            continue

        params = default_configuration_settings
        parent = None
        last_key = None

        # Get the parameters for testing
        for step in config_path:
            parent = params
            last_key = step
            params = params.get(step, "_failed_to_fetch")
            if params == "_failed_to_fetch":
                raise ValueError(f"Failed to fetch data from config path: {config_path}")

        # Add root data for Multiwfn
        if config_path[0] == "multiwfn":
            for key, value in default_configuration_settings["multiwfn"].items():
                if type(value) != dict:
                    params[key] = value

        # Delete the configuration path after the parameters have been fetched
        if parent is not None and last_key is not None:
            del parent[last_key]

        # Validate the parameters
        params["feature_info"] = feature_information
        assert type(params) == dict

        validated_params = config_data_validator(
            config_path=config_path, params=params, _namespace="irrelevant"
        )
        assert type(validated_params) == dict

        del params["feature_info"]
        assert set(validated_params.keys()) == set(params.keys())

    # Ensure that default_configuration_settings was fully swept, the remaining keys and
    # values are expected and correct
    assert default_configuration_settings == {
        "bonafide": {},
        "dscribe": {},
        "morfeus": {},
        "multiwfn": {"OMP_STACKSIZE": "1G", "NUM_THREADS": 4},
        "rdkit": {},
    }

    # Ensure that only INFO messages were logged
    assert len(caplog.records) > 0
    assert all(record.levelno == logging.INFO for record in caplog.records)


##########################
# Test wrong data types. #
##########################


@pytest.mark.wrong_data_types
def test_wrong_data_types(
    caplog: pytest.LogCaptureFixture,
    all_config_paths: List[List[str]],
    default_configuration_settings: Dict[str, Dict[str, Any]],
    feature_information: Dict[int, Dict[str, Any]],
):
    """Test for wrong data types in the configuration settings."""
    # Loop over sections
    for config_path in all_config_paths:
        params = default_configuration_settings
        # Get the parameters
        for step in config_path:
            params = params.get(step, "_failed_to_fetch")
            if params == "_failed_to_fetch":
                raise ValueError(f"Failed to fetch data from config path: {config_path}")

        # Add root data for Multiwfn
        if config_path[0] == "multiwfn":
            for key, value in default_configuration_settings["multiwfn"].items():
                if type(value) != dict:
                    params[key] = value

        # Loop over parameters and modify data types
        params_ = copy.deepcopy(params)
        _data_modified = False
        for key, value in params_.items():
            if type(value) in [int, float]:
                _data_modified = True
                params_[key] = "not_a_number"
            elif type(value) == list:
                _data_modified = True
                params_[key] = "not_a_list"
            elif type(value) == str:
                _data_modified = True
                params_[key] = [value]
            elif type(value) == bool:
                _data_modified = True
                params_[key] = str(value)

        # Validate the parameters
        params_["feature_info"] = feature_information

        if _data_modified:
            with pytest.raises(ValueError):
                _ = config_data_validator(
                    config_path=config_path, params=params_, _namespace="irrelevant"
                )
            assert any(record.levelno == logging.ERROR for record in caplog.records)
        else:
            validated_params = config_data_validator(
                config_path=config_path, params=params_, _namespace="irrelevant"
            )
            assert type(validated_params) == dict
            assert len(caplog.records) > 0
            assert all(record.levelno == logging.INFO for record in caplog.records)

        # Reset log records for next iteration
        caplog.clear()


# ##########################################
# # Test missing or additional parameters. #
# ##########################################


@pytest.mark.wrong_param_set
@pytest.mark.parametrize(
    "config_path, to_be_removed",
    [
        (["alfabet"], "python_interpreter_path"),
        (["bonafide", "autocorrelation"], "depth"),
        (["dscribe", "lmbtr"], "weighting_function"),
        (["xtb"], "method"),
        (["qmdesc"], None),
        (["multiwfn", "fuzzy"], "n_iterations_becke_partition"),
    ],
)
def test_wrong_param_set(
    caplog: pytest.LogCaptureFixture,
    config_path: List[str],
    to_be_removed: str,
    default_configuration_settings: Dict[str, Dict[str, Any]],
    feature_information: Dict[int, Dict[str, Any]],
) -> None:
    # Get the parameters
    params = default_configuration_settings
    for step in config_path:
        params = params.get(step, "_failed_to_fetch")
        if params == "_failed_to_fetch":
            raise ValueError(f"Failed to fetch data from config path: {config_path}")

    # Add root data for Multiwfn
    if config_path[0] == "multiwfn":
        for key, value in default_configuration_settings["multiwfn"].items():
            if type(value) != dict:
                params[key] = value

    # Add a parameter (will simply be ignored by pydantic and will be not part of the validated
    # parameters)
    params["feature_info"] = feature_information
    params_added = copy.deepcopy(params)
    params_added["I_should_really_not_be_here"] = True
    validated_params = config_data_validator(
        config_path=config_path, params=params_added, _namespace="irrelevant"
    )
    assert type(validated_params) == dict

    del params["feature_info"]
    assert set(validated_params.keys()) == set(params.keys())

    # Remove a parameter (will raise an error)
    if to_be_removed is None:
        return

    params["feature_info"] = feature_information
    params_removed = copy.deepcopy(params)
    del params_removed[to_be_removed]

    with pytest.raises(ValueError):
        _ = config_data_validator(
            config_path=config_path, params=params_removed, _namespace="irrelevant"
        )
    assert any(record.levelno == logging.ERROR for record in caplog.records)


# #########################################################
# # Test specific parameters and not implemented origins. #
# #########################################################


@pytest.mark.specific_param
@pytest.mark.parametrize(
    "config_path, param_name, new_value, is_valid",
    [
        (["dscribe", "soap"], "species", "all", False),
        (["dscribe", "soap"], "species", False, False),
        (["dscribe", "soap"], "species", ["H", "C", "N", "O", "F", "Cl", "Bb"], False),
        (["dscribe", "soap"], "species", ["H", "C", "N", "O", "F", "Cl"], True),
        (["bonafide", "functional_group"], "key_level", "l4", False),
        (["bonafide", "functional_group"], "key_level", "L1", True),
        (["kallisto"], "size", [5, 2], False),
        (["kallisto"], "size", [2, 3, 4], False),
        (["kallisto"], "size", "2,3", False),
        (["morfeus", "buried_volume"], "radius", 0, False),
        (["morfeus", "buried_volume"], "radii_type", "ALVARez", True),
        (["psi4"], "solvent", "1,2-dimethoxyethane", False),
        (["psi4"], "solvent", "WATER", True),
        (["xtb"], "etemp_native", 4523.3, False),  # temp must be int
        # No validation class implemented for origin
        (["MissingValidator"], "size", 5.0, True),
        (["MissingValidator2", "option"], "scale", False, True),
        (["MissingValidatorEmpty"], "", None, True),
    ],
)
def test_specific_param(
    caplog: pytest.LogCaptureFixture,
    config_path: List[str],
    param_name: str,
    new_value: Any,
    is_valid: bool,
    default_configuration_settings: Dict[str, Dict[str, Any]],
    feature_information: Dict[int, Dict[str, Any]],
):
    # Get the parameters
    _misses_validator = False
    if config_path[0] in ["MissingValidator", "MissingValidator2", "MissingValidatorEmpty"]:
        _misses_validator = True
        if param_name != "":
            params = {param_name: new_value}
        else:
            params = {}
    else:
        params = default_configuration_settings
        for step in config_path:
            params = params.get(step, "_failed_to_fetch")
            if params == "_failed_to_fetch":
                raise ValueError(f"Failed to fetch data from config path: {config_path}")

    # Add root data for Multiwfn
    if config_path[0] == "multiwfn":
        for key, value in default_configuration_settings["multiwfn"].items():
            if type(value) != dict:
                params[key] = value

    params["feature_info"] = feature_information

    # Change parameter value
    params[param_name] = new_value

    # Do validation (parameter set to invalid value)
    if not is_valid:
        with pytest.raises(ValueError):
            _ = config_data_validator(
                config_path=config_path, params=params, _namespace="irrelevant"
            )
        assert any(record.levelno == logging.ERROR for record in caplog.records)
        return

    # Do validation (parameter set to valid value)
    validated_params = config_data_validator(
        config_path=config_path, params=params, _namespace="irrelevant"
    )
    assert type(validated_params) == dict

    del params["feature_info"]
    assert set(validated_params.keys()) == set(params.keys())

    # Check logging depending on if a known or unknown origin was passed
    assert len(caplog.records) > 0
    if _misses_validator is False:
        assert all(record.levelno == logging.INFO for record in caplog.records)
    else:
        assert any(record.levelno == logging.WARNING for record in caplog.records)
