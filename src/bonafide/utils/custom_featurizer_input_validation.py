"""Type and format validation of the dictionary provided by the user for custom featurizers."""

import logging
from typing import Any, Dict, Tuple

from bonafide.utils.constants import (
    DATA_TYPES,
    DIMENSIONALITIES,
    FEATURE_TYPES,
)
from bonafide.utils.helper_functions import standardize_string


def custom_featurizer_data_validator(
    custom_metadata: Dict[str, Any],
    feature_info: Dict[int, Dict[str, Any]],
    feature_config: Dict[str, Any],
    namespace: str,
    loc: str,
) -> Tuple[str, Dict[str, Any]]:
    """Validate the user input for introducing a custom featurizer to BONAFIDE.

    Parameters
    ----------
    custom_metadata : Dict[str, Any]
        The dictionary with the required metadata for the custom featurizer.
    feature_info : Dict[int, Dict[str, Any]]
        The metadata of all implemented atom and bond features, e.g., the name of the feature, its
        dimensionality requirements (either 2D or 3D), or the program it is calculated with
        (origin).
    feature_config : Dict[str, Any]
        The configuration settings for the individual programs used for feature calculation.
    namespace : str
        The namespace for the molecule as defined by the user when reading in the molecule.
    loc : str
        The location string representing the current class and method for logging purposes.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        A tuple containing the origin string of the custom featurizer and the validated metadata
        dictionary.
    """
    # Check if required keys were provided
    required_keys = list(feature_info[list(feature_info.keys())[0]].keys())
    required_keys_ = [x for x in required_keys]
    required_keys.sort()
    provided_keys = list(custom_metadata.keys())
    provided_keys.sort()
    if required_keys != provided_keys:
        _errmsg = (
            f"Invalid input to 'custom_metadata': provided data format is incorrect. "
            f"The (only) required keys of the input dictionary are {required_keys_}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise KeyError(f"{loc}(): {_errmsg}")

    # Check name
    _inpt = type(custom_metadata["name"])
    if _inpt != str:
        _errmsg = (
            "Invalid input to 'name' in 'custom_metadata': must be of type str but obtained "
            f"{_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise TypeError(f"{loc}(): {_errmsg}")

    name_ = standardize_string(inp_data=custom_metadata["name"])
    if len(name_) == 0:
        _errmsg = "Invalid input to 'name' in 'custom_metadata': must not be an empty string."
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise ValueError(f"{loc}(): {_errmsg}")

    custom_metadata["name"] = custom_metadata["name"].strip()

    # Check origin
    _inpt = type(custom_metadata["origin"])
    if _inpt != str:
        _errmsg = (
            "Invalid input to 'origin' in 'custom_metadata': must be of type str but obtained "
            f"{_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise TypeError(f"{loc}(): {_errmsg}")

    origin_ = standardize_string(inp_data=custom_metadata["origin"])
    if origin_ in list(feature_config.keys()):
        _errmsg = f"Invalid input to 'origin' in 'custom_metadata': '{origin_}' is already in use."
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise ValueError(f"{loc}(): {_errmsg}")

    custom_metadata["origin"] = origin_

    # Check feature_type
    _inpt = type(custom_metadata["feature_type"])
    if _inpt != str:
        _errmsg = (
            "Invalid input to 'feature_type' in 'custom_metadata': must be of type str but "
            f"obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise TypeError(f"{loc}(): {_errmsg}")

    feature_type_ = standardize_string(inp_data=custom_metadata["feature_type"])
    if feature_type_ not in FEATURE_TYPES:
        _errmsg = (
            f"Invalid input to 'feature_type' in 'custom_metadata': '{feature_type_}' is "
            f"not supported, available: {FEATURE_TYPES}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise ValueError(f"{loc}(): {_errmsg}")

    custom_metadata["feature_type"] = feature_type_

    # Check dimensionality
    _inpt = type(custom_metadata["dimensionality"])
    if _inpt != str:
        _errmsg = (
            "Invalid input to 'dimensionality' in 'custom_metadata': must be of type str but "
            f"obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise TypeError(f"{loc}(): {_errmsg}")

    dimensionality_ = standardize_string(inp_data=custom_metadata["dimensionality"], case="upper")
    if dimensionality_ not in DIMENSIONALITIES:
        _errmsg = (
            f"Invalid input to 'dimensionality' in 'custom_metadata': '{dimensionality_}' is "
            f"not supported, available: {DIMENSIONALITIES}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise ValueError(f"{loc}(): {_errmsg}")

    custom_metadata["dimensionality"] = dimensionality_

    # Check data_type
    _inpt = type(custom_metadata["data_type"])
    if _inpt != str:
        _errmsg = (
            "Invalid input to 'data_type' in 'custom_metadata': must be of type str but obtained "
            f"{_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise TypeError(f"{loc}(): {_errmsg}")

    data_type_ = standardize_string(inp_data=custom_metadata["data_type"])
    if data_type_ not in DATA_TYPES:
        _errmsg = (
            f"Invalid input to 'data_type' in 'custom_metadata': '{data_type_}' is not supported, "
            f"available: {DATA_TYPES}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise ValueError(f"{loc}(): {_errmsg}")

    custom_metadata["data_type"] = data_type_

    # Check requires_electronic_structure_data
    _inpt = type(custom_metadata["requires_electronic_structure_data"])
    if _inpt != bool:
        _errmsg = (
            "Invalid input to 'requires_electronic_structure_data' in 'custom_metadata': must be "
            f"of type bool but obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise TypeError(f"{loc}(): {_errmsg}")

    # Check requires_bond_data
    _inpt = type(custom_metadata["requires_bond_data"])
    if _inpt != bool:
        _errmsg = (
            "Invalid input to 'requires_bond_data' in 'custom_metadata': must be "
            f"of type bool but obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise TypeError(f"{loc}(): {_errmsg}")

    # Check requires_charge
    _inpt = type(custom_metadata["requires_charge"])
    if _inpt != bool:
        _errmsg = (
            "Invalid input to 'requires_charge' in 'custom_metadata': must be "
            f"of type bool but obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise TypeError(f"{loc}(): {_errmsg}")

    # Check requires_multiplicity
    _inpt = type(custom_metadata["requires_multiplicity"])
    if _inpt != bool:
        _errmsg = (
            "Invalid input to 'requires_multiplicity' in 'custom_metadata': must be "
            f"of type bool but obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise TypeError(f"{loc}(): {_errmsg}")

    # Check config_path
    _inpt = type(custom_metadata["config_path"])
    if _inpt != dict:
        _errmsg = (
            "Invalid input to 'config_path' in 'custom_metadata': must be of type dict but "
            f"obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise TypeError(f"{loc}(): {_errmsg}")

    # Check factory
    _inpt = type(custom_metadata["factory"])
    if _inpt != type:
        _errmsg = (
            "Invalid input to 'factory' in 'custom_metadata': must be pointing to a class but "
            f"obtained {_inpt.__name__}."
        )
        logging.error(f"'{namespace}' | {loc}()\n{_errmsg}")
        raise TypeError(f"{loc}(): {_errmsg}")

    return origin_, custom_metadata
