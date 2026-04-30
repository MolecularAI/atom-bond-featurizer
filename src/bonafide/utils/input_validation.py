"""Type and format validation of the configuration settings parameters of the individual
featurizers."""

from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from bonafide.utils.constants import (
    ATOMIC_RADII_MULTIWFN_POPULATION,
    AVERAGE_METHODS_DSCRIBE_SOAP,
    CNTYPE_METHODS_KALLISTO,
    DISTAL_VOLUME_METHODS_MORFEUS_BV,
    EEM_PARAMETERS_MULTIWFN_POPULATION,
    ELECTRONEGATIVITY_EN_SCALES,
    ELEMENT_SYMBOLS,
    ESP_TYPE_MULTIWFN_POPULATION,
    FUNCTIONAL_GROUP_KEY_LEVELS,
    GEOMETRY_FUNCTION_METHODS_DSCRIBE_LMBTR,
    IBIS_GRID_METHODS_MULTIWFN_BOND_ANALYSIS,
    IGM_TYPES_MULTIWFN_BOND_ANALYSIS,
    INTEGRATION_GRID_METHODS_MULTIWFN_FUZZY,
    ITERABLE_OPTIONS_MULTIWFN_CDFT,
    METHOD_METHODS_MENDELEEV,
    METHODS_MORFEUS_LOCAL_FORCE,
    METHODS_XTB,
    NORMALIZATION_METHODS_DSCRIBE_LMBTR,
    PARTITION_SCHEME_METHODS_MULTIWFN_FUZZY,
    PYRAMIDALIZATION_CALCULATION_METHODS_MORFEUS_PYRAMIDALIZATION,
    RADII_TYPE_METHODS_MORFEUS_BV_CONE_SOLID_ANGLE,
    RADII_TYPE_METHODS_MORFEUS_DISPERSION,
    RADII_TYPE_METHODS_MORFEUS_SASA,
    RADIUS_BECKE_PARTITION_METHODS_MULTIWFN_FUZZY,
    RADIUS_BECKE_PARTITION_METHODS_MULTIWFN_POPULATION,
    RBF_METHODS_DSCRIBE_SOAP,
    REAL_SPACE_FUNCTIONS_MULTIWFN,
    SOLVENT_MODEL_SOLVERS_PSI4,
    SOLVENT_MODELS_XTB,
    SOLVENTS_PSI4,
    SOLVENTS_XTB,
    VDWTYPE_METHODS_KALLISTO,
    WEIGHTING_FUNCTION_METHODS_DSCRIBE_LMBTR,
)
from bonafide.utils.helper_functions import get_function_or_method_name


class _StandardizeStrMixin:
    """Standardize string inputs before validation."""

    @field_validator("*", mode="before")
    @classmethod
    def standardize_strings(cls, value: Any, info: ValidationInfo) -> Any:
        """Standardize string inputs by stripping whitespace and converting to lowercase.

        If the value is not a string or the field name is in a predefined blacklist, it is returned
        as is (with whitespaces stripped if it is a string).

        Parameters
        ----------
        value : Any
            The value to be standardized.
        info : ValidationInfo
            Information about the field being validated.

        Returns
        -------
        Any
            The standardized value if it is a string, otherwise the original value.
        """
        _black_list = ["XTBHOME", "PSI_SCRATCH"]
        _dtype = type(value)
        if info.field_name in _black_list and _dtype == str:
            return value.strip()
        if _dtype == str:
            return value.strip().lower()
        return value


class _ValidateSpeciesMixin:
    """Validate a list of chemical element symbols."""

    @field_validator("species", mode="before")
    @classmethod
    def validate_species_before(cls, value: Any) -> List[str]:
        """Validate ``species`` before type validation.

        "auto" is the only valid string input.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        List[str]
            List of element symbols or ["auto"] if the input is valid.
        """
        _errmsg = "Input must be either 'auto' or a list of element symbols to consider"
        if type(value) == str:
            if value == "auto":
                return [value]
            raise PydanticCustomError("", _errmsg)
        elif type(value) == list:
            return value
        else:
            raise PydanticCustomError("", _errmsg)

    @field_validator("species", mode="after")
    @classmethod
    def validate_species_after(cls, value: List[str]) -> Union[str, List[str]]:
        """Validate ``species`` after type validation.

        Parameters
        ----------
        value : List[str]
            The list of element symbols to be validated.

        Returns
        -------
        Union[str, List[str]]
            Returns "auto" if the input is ["auto"], otherwise returns the validated list of
            chemical element symbols.
        """
        if value == ["auto"]:
            return value[0]

        for symbol in value:
            if symbol not in ELEMENT_SYMBOLS:
                _errmsg = f"Input must only contain {ELEMENT_SYMBOLS}"
                raise PydanticCustomError("", _errmsg)
        return value


class _ValidateIterableIntOptionMixin:
    """Mixin to validate the input of a feature index corresponding to a feature of data type int
    or float.
    """

    feature_info: Dict[int, Dict[str, Any]]
    iterable_option: List[Any]

    @field_validator("iterable_option", mode="before")
    @classmethod
    def validate_iterable_option_before(cls, value: Any) -> Any:
        """Validate ``iterable_option`` before type validation.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        Any
            The validated input list. If the input is a single integer, it is converted to a list.
        """
        if any([value is None, value == "", value == []]):
            _errmsg = "Input must not be empty"
            raise PydanticCustomError("", _errmsg)
        elif type(value) == int:
            value = [value]
        return value

    @model_validator(mode="after")
    def check_iterable_option(self) -> _ValidateIterableIntOptionMixin:
        """Validate ``iterable_option`` after type validation.

        Returns
        -------
        _ValidateIterableIntOptionMixin
            The instance with the validated and formatted iterable option.
        """
        _new_iterable_option_list = []
        for idx in self.iterable_option:
            # Check if the iterable option is a valid feature index
            if idx not in self.feature_info:
                _errmsg = (
                    f"Input is not a valid feature index, obtained: {idx} "
                    f"(of type '{type(idx).__name__}')"
                )
                raise PydanticCustomError("", _errmsg)

            # Check if the index corresponds to an atom feature of data type float or int
            _feature_name = self.feature_info[idx]["name"]
            _feature_type = self.feature_info[idx]["feature_type"]
            _data_type = self.feature_info[idx]["data_type"]
            if _feature_type != "atom" or _data_type not in ["float", "int"]:
                _errmsg = (
                    "Input is not a feature index corresponding to an atom feature of type "
                    f"'int' or 'float', obtained: {idx} (of type '{type(idx).__name__}')"
                )
                raise PydanticCustomError("", _errmsg)
            _new_iterable_option_list.append(_feature_name)

        # Replace the feature indices in the iterable options list with the feature names
        self.iterable_option = [x for x in _new_iterable_option_list]

        return self


class ValidateAlfabet(BaseModel):
    """Validate the configuration settings for the alfabet features."""

    # Don't standardize the string to avoid changing the path
    python_interpreter_path: StrictStr


class ValidateBonafideAutocorrelation(_ValidateIterableIntOptionMixin, BaseModel):
    """Validate the configuration settings for the autocorrelation features.

    Attributes
    ----------
    depth : StrictInt
        The depth of the autocorrelation, must be a positive integer.
    iterable_option : List[StrictInt]
        A list of feature indices to be used for the autocorrelation calculation.
    feature_info : Dict
        A dictionary containing information about the available features, where keys are feature
        indices and values are dictionaries with feature details.
    """

    depth: StrictInt = Field(gt=0)
    iterable_option: List[StrictInt]
    feature_info: Dict[int, Dict[str, Any]]


class ValidateBonafideConstant(BaseModel):
    """Validate the configuration settings for the constant atom/bond features.

    Attributes
    ----------
    atom_constant : StrictStr
        The constant value to be assigned the requested atoms.
    bond_constant : StrictStr
        The constant value to be assigned the requested bonds.
    """

    # Don't standardize strings to avoid overwriting the custom user input
    atom_constant: StrictStr
    bond_constant: StrictStr


class ValidateBonafideDistance(BaseModel):
    """Validate the configuration settings for the distance-based features.

    Attributes
    ----------
    n_bonds_cutoff : StrictInt
        The number of bonds to consider for the feature calculation as a distance cutoff.
    radius_cutoff : StrictFloat
        The radius in Angstrom to consider for the feature calculation as a distance cutoff.
    """

    n_bonds_cutoff: StrictInt = Field(gt=0)
    radius_cutoff: StrictFloat = Field(gt=0)


class ValidateBonafideFunctionalGroup(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the functional group features.

    Attributes
    ----------
    key_level : StrictStr
        The key level for the functional group features which determines how fine-grained the
        analysis is carried out.
    custom_groups : List[List[StrictStr]]
        A list of custom functional groups defined by the user, where each functional group is
        represented by a list containing the name of the functional group and its corresponding
        SMARTS pattern.
    """

    key_level: StrictStr
    custom_groups: List[List[StrictStr]]

    @field_validator("key_level")
    @classmethod
    def validate_key_level(cls, value: str) -> str:
        """Validate ``key_level``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The formatted and validated key level.
        """
        if value not in FUNCTIONAL_GROUP_KEY_LEVELS:
            _errmsg = f"Input must be one of {FUNCTIONAL_GROUP_KEY_LEVELS}"
            raise PydanticCustomError("", _errmsg)
        return value

    @field_validator("custom_groups")
    @classmethod
    def validate_custom_groups(cls, value: List[List[str]]) -> List[List[str]]:
        """Validate ``custom_groups``.

        Parameters
        ----------
        value : List[List[str]]
            The value to be validated.

        Returns
        -------
        List[List[str]]
            The validated list of custom functional groups.
        """
        for custom_group in value:
            if len(custom_group) != 2:
                _errmsg = (
                    "Each custom functional group must be a list of length 2, with the first entry "
                    "being the name of the functional group and the second entry being the SMARTS "
                    "pattern defining the group"
                )
                raise PydanticCustomError("", _errmsg)
        return value


class ValidateBonafideOxidationState(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the oxidation state feature.

    Attributes
    ----------
    en_scale : StrictStr
        The name of the electronegativity scale to be used for the oxidation state calculation.
    """

    en_scale: StrictStr

    @field_validator("en_scale")
    @classmethod
    def validate_en_scale(cls, value: str) -> str:
        """Validate ``en_scale``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated electronegativity scale.
        """
        if value not in ELECTRONEGATIVITY_EN_SCALES:
            _errmsg = f"Input must be one of {ELECTRONEGATIVITY_EN_SCALES}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateBonafideSymmetry(BaseModel):
    """Validate the configuration settings for the symmetry feature.

    For further details, please refer to the RDKit documentation
    (https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html, last accessed on
    14.10.2025).

    Attributes
    ----------
    reduce_to_canonical : StrictBool
        Whether to calculate features only for the first of the symmetry-equivalent atoms in the
        canonical rank atom list.
    includeChirality : StrictBool
        Whether to include chirality information when calculating the symmetry feature.
    includeIsotopes : StrictBool
        Whether to consider isotopes when calculating the symmetry feature.
    includeAtomMaps : StrictBool
        Whether to include atom mapping numbers when calculating the symmetry feature.
    includeChiralPresence : StrictBool
        Whether to include the presence of chiral centers when calculating the symmetry feature.
    consider_resonance : StrictBool
        Whether to consider resonance forms of the molecule when finding out which atoms are
        symmetric to each other.
    resonance_ALLOW_CHARGE_SEPARATION : StrictBool
        Whether to allow resonance forms with charge separation when considering resonance forms of
        the molecule.
    resonance_ALLOW_INCOMPLETE_OCTETS : StrictBool
        Whether to allow resonance forms with incomplete octets when considering resonance forms of
        the molecule.
    resonance_KEKULE_ALL : StrictBool
        Whether to generate all possible Kekule resonance forms when considering resonance forms of
        the molecule.
    resonance_UNCONSTRAINED_ANIONS : StrictBool
        Whether to allow unconstrained anions when considering resonance forms of the molecule.
    resonance_UNCONSTRAINED_CATIONS : StrictBool
        Whether to allow unconstrained cations when considering resonance forms of the molecule.
    """

    reduce_to_canonical: StrictBool
    includeChirality: StrictBool
    includeIsotopes: StrictBool
    includeAtomMaps: StrictBool
    includeChiralPresence: StrictBool
    consider_resonance: StrictBool
    resonance_ALLOW_CHARGE_SEPARATION: StrictBool
    resonance_ALLOW_INCOMPLETE_OCTETS: StrictBool
    resonance_KEKULE_ALL: StrictBool
    resonance_UNCONSTRAINED_ANIONS: StrictBool
    resonance_UNCONSTRAINED_CATIONS: StrictBool


class ValidateDbstep(BaseModel):
    """Validate the configuration settings for the dbstep features.

    For further details, please refer to the dbstep repository (https://github.com/patonlab/DBSTEP,
    last accessed on 05.09.2025).

    Attributes
    ----------
    r : StrictFloat
        The cutoff radius, must be a positive float.
    scan : List[StrictFloat]
        A list of three values defining the scan range and step size.
    exclude : List[StrictInt]
        A list of atom indices to be excluded from the feature calculation.
    noH : StrictBool
        Whether to exclude hydrogen atoms from the feature calculation.
    addmetals : StrictBool
        Whether to include metal atoms in the feature calculation.
    grid : StrictFloat
        The grid point spacing, must be a positive float.
    vshell : StrictBool
        Whether to calculate the buried volume of a hollow sphere.
    scalevdw : StrictFloat
        The scaling factor for van-der-Waals radii, must be a positive float.
    """

    r: StrictFloat = Field(gt=0)
    scan: List[StrictFloat]
    exclude: List[StrictInt]
    noH: StrictBool
    addmetals: StrictBool
    grid: StrictFloat = Field(gt=0)
    vshell: StrictBool
    scalevdw: StrictFloat = Field(gt=0)

    @field_validator("scan")
    @classmethod
    def validate_scan(cls, value: List[float]) -> Union[str, bool]:
        """Validate ``scan``.

        Parameters
        ----------
        value : List[float]
            The value to be validated.

        Returns
        -------
        Union[str, bool]
            The validated and formatted scan range and step size, or ``False`` if the input is
            empty.
        """
        if len(value) != 0:
            if len(value) != 3:
                _errmsg = "Input must contain exactly 3 values if not left empty"
                raise PydanticCustomError("", _errmsg)
            value_str = ":".join([str(val) for val in value])
            return value_str

        return False

    @field_validator("exclude")
    @classmethod
    def validate_exclude(cls, value: List[int]) -> Union[str, bool]:
        """Validate ``exclude``.

        Parameters
        ----------
        value : List[int]
            The value to be validated.

        Returns
        -------
        Union[str, bool]
            The validated and formatted list of atom indices to be excluded, or ``False`` if the
            input is empty.
        """
        if value == []:
            return False
        value_str = ",".join([str(val) for val in value])
        return value_str


class ValidateDscribeAcsf(_ValidateSpeciesMixin, BaseModel):
    """Validate the configuration settings for the dscribe atom-centered symmetry functions
    feature.

    For further details, please refer to the dscribe documentation
    (https://singroup.github.io/dscribe/0.3.x/index.html, last accessed on 05.09.2025).

    Attributes
    ----------
    r_cut : StrictFloat
        The smooth cutoff radius, must be a positive float.
    species : List[StrictStr]
        A list of chemical element symbols to be considered in the feature calculation.
    g2_params : List[List[StrictFloat]]
        The parameters for the G2 symmetry functions.
    g3_params : List[StrictFloat]
        The parameters for the G3 symmetry functions.
    g4_params : List[List[StrictFloat]]
        The parameters for the G4 symmetry functions.
    g5_params : List[List[StrictFloat]]
        The parameters for the G5 symmetry functions.
    """

    r_cut: StrictFloat = Field(gt=0)
    species: List[StrictStr]
    g2_params: Any
    g3_params: Any
    g4_params: Any
    g5_params: Any

    @field_validator("g2_params", "g3_params", "g4_params", "g5_params")
    @classmethod
    def validate_params(cls, value: Any, info: ValidationInfo) -> Optional[Any]:
        """Validate ``g2_params``, ``g3_params``, ``g4_params``, and ``g5_params``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        Any
            The validated value, either ``None`` or the value specified by the user.
        """
        if type(value) == str:
            if value.strip().lower() == "none":
                return None
            else:
                _errmsg = "Input must be of type list or 'none'"
                raise PydanticCustomError("", _errmsg)

        if type(value) != list:
            _errmsg = "Input must be of type list or 'none'"
            raise PydanticCustomError("", _errmsg)

        if len(value) == 0:
            _errmsg = "Input must not be an empty list"
            raise PydanticCustomError("", _errmsg)

        for idx1, el in enumerate(value):
            if type(el) != list and info.field_name != "g3_params":
                _errmsg = (
                    f"Each entry in the list must be of type list, but obtained {type(el).__name__}"
                )
                raise PydanticCustomError("", _errmsg)

            if info.field_name == "g3_params":
                try:
                    if type(el) in [int, float]:
                        _f = float(el)
                    else:
                        raise Exception()
                except Exception:
                    _errmsg = (
                        "Each entry in the list must be of type int or float, but obtained "
                        f"{type(el).__name__}"
                    )
                    raise PydanticCustomError("", _errmsg)
                else:
                    value[idx1] = _f

            if info.field_name == "g2_params":
                if len(el) != 2:
                    _errmsg = "Each inner list must contain exactly 2 entries"
                    raise PydanticCustomError("", _errmsg)

            if info.field_name in ["g4_params", "g5_params"]:
                if len(el) != 3:
                    _errmsg = "Each inner list must contain exactly 3 entries"
                    raise PydanticCustomError("", _errmsg)

            if info.field_name in ["g2_params", "g4_params", "g5_params"]:
                for idx2, val in enumerate(el):
                    try:
                        if type(val) in [int, float]:
                            _f = float(val)
                        else:
                            raise Exception()
                    except Exception:
                        _errmsg = (
                            "Each entry in the inner list must be of type int or float, but "
                            f"obtained {type(val).__name__}"
                        )
                        raise PydanticCustomError("", _errmsg)
                    else:
                        value[idx1][idx2] = _f

        return value


class ValidateDscribeCoulombMatrix(BaseModel):
    """Validate the configuration settings for the dscribe Coulomb matrix-based feature.

    For further details, please refer to the dscribe documentation
    (https://singroup.github.io/dscribe/0.3.x/index.html, last accessed on 05.09.2025).

    Attributes
    ----------
    scaling_exponent : StrictFloat
        The exponent used for the distance scaling.
    """

    scaling_exponent: StrictFloat


class ValidateDscribeLmbtr(_StandardizeStrMixin, _ValidateSpeciesMixin, BaseModel):
    """Validate the configuration settings for the dscribe local many-body tensor representation
    feature.

    For further details, please refer to the dscribe documentation
    (https://singroup.github.io/dscribe/0.3.x/index.html, last accessed on 05.09.2025).

    Attributes
    ----------
    species : List[StrictStr]
        A list of chemical element symbols to be considered in the feature calculation.
    geometry_function : StrictStr
        The name of the geometry function.
    grid_min : StrictFloat
        The minimum value of the grid, must be a float.
    grid_max : StrictFloat
        The maximum value of the grid, must be a float.
    grid_sigma : StrictFloat
        The width of the Gaussian functions, must be a positive float.
    grid_n : StrictFloat
        The number of grid points, must be a non-negative integer.
    weighting_function : StrictStr
        The name of the weighting function.
    weighting_scale : StrictFloat
        The scaling factor of the weighting function, must be a float.
    weighting_threshold : StrictFloat
        The threshold of the weighting function, must be a positive float.
    normalize_gaussians : StrictBool
        Whether to normalize the Gaussians to an area of 1.
    normalization : StrictStr
        The normalization method.
    """

    species: List[StrictStr]
    geometry_function: StrictStr
    grid_min: StrictFloat
    grid_max: StrictFloat
    grid_sigma: StrictFloat
    grid_n: StrictFloat = Field(ge=0)
    weighting_function: StrictStr
    weighting_scale: StrictFloat
    weighting_threshold: StrictFloat = Field(gt=0)
    normalize_gaussians: StrictBool
    normalization: StrictStr

    @field_validator("geometry_function")
    @classmethod
    def validate_geometry_function(cls, value: str) -> str:
        """Validate ``geometry_function``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated geometry function.
        """
        if value not in GEOMETRY_FUNCTION_METHODS_DSCRIBE_LMBTR:
            _errmsg = f"Input must be one of {GEOMETRY_FUNCTION_METHODS_DSCRIBE_LMBTR}"
            raise PydanticCustomError("", _errmsg)
        return value

    @field_validator("weighting_function")
    @classmethod
    def validate_weighting_function(cls, value: str) -> str:
        """Validate ``weighting_function``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated weighting function.
        """
        if value not in WEIGHTING_FUNCTION_METHODS_DSCRIBE_LMBTR:
            _errmsg = f"Input must be one of {WEIGHTING_FUNCTION_METHODS_DSCRIBE_LMBTR}"
            raise PydanticCustomError("", _errmsg)
        return value

    @field_validator("normalization")
    @classmethod
    def validate_normalization(cls, value: str) -> str:
        """Validate ``normalization``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated normalization method.
        """
        if value not in NORMALIZATION_METHODS_DSCRIBE_LMBTR:
            _errmsg = f"Input must be one of {NORMALIZATION_METHODS_DSCRIBE_LMBTR}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateDscribeSoap(_StandardizeStrMixin, _ValidateSpeciesMixin, BaseModel):
    """Validate the configuration settings for the dscribe smooth overlap of atomic positions
    feature.

    For further details, please refer to the dscribe documentation
    (https://singroup.github.io/dscribe/0.3.x/index.html, last accessed on 05.09.2025).

    Attributes
    ----------
    r_cut : StrictFloat
        The cutoff to define the local environment, must be a positive float.
    n_max : StrictInt
        The number of radial basis functions, must be a positive integer.
    l_max : StrictInt
        The maximum degree of spherical harmonics, must be a non-negative integer.
    species : List[StrictStr]
        A list of chemical element symbols to be considered in the feature calculation.
    sigma : StrictFloat
        The width of the Gaussian functions, must be a positive float.
    rbf : StrictStr
        The radial basis function.
    average : StrictStr
        The averaging method.
    """

    r_cut: StrictFloat
    n_max: StrictInt = Field(gt=0)
    l_max: StrictInt = Field(ge=0)
    species: List[StrictStr]
    sigma: StrictFloat = Field(gt=0)
    rbf: StrictStr
    average: StrictStr

    @field_validator("rbf")
    @classmethod
    def validate_rbf(cls, value: str) -> str:
        """Validate ``rbf``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated radial basis function.
        """
        if value not in RBF_METHODS_DSCRIBE_SOAP:
            _errmsg = f"Input must be one of {RBF_METHODS_DSCRIBE_SOAP}"
            raise PydanticCustomError("", _errmsg)
        return value

    @field_validator("average")
    @classmethod
    def validate_average(cls, value: str) -> str:
        """Validate ``average``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated averaging method.
        """
        if value not in AVERAGE_METHODS_DSCRIBE_SOAP:
            _errmsg = f"Input must be one of {AVERAGE_METHODS_DSCRIBE_SOAP}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateKallisto(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Kallisto features.

    For further details, please refer to the Kallisto documentation
    (https://ehjc.gitbook.io/kallisto/, last accessed on 05.09.2025).

    Attributes
    ----------
    cntype : StrictStr
        The name of the coordination number calculation method.
    size : List[StrictInt]
        The definition of the proximity shell.
    vdwtype : StrictStr
        The name of the method to define reference van-der-Waals radii.
    angstrom : StrictBool
        Whether to calculate van-der-Waals radii in Angstrom.
    """

    cntype: StrictStr
    size: List[StrictInt]
    vdwtype: StrictStr
    angstrom: StrictBool

    @field_validator("cntype")
    @classmethod
    def validate_cntype(cls, value: str) -> str:
        """Validate ``cntype``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated coordination number method.
        """
        if value not in CNTYPE_METHODS_KALLISTO:
            _errmsg = f"Input must be one of {CNTYPE_METHODS_KALLISTO}"
            raise PydanticCustomError("", _errmsg)
        return value

    @field_validator("size", mode="before")
    @classmethod
    def validate_size_before(cls, value: Any) -> List[int]:
        """Validate ``size`` before type validation.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        List[int]
            The validated definition of the proximity shell.
        """
        _errmsg = "Input must be a list consisting exactly of two integer numbers"
        if type(value) != list:
            raise PydanticCustomError("", _errmsg)
        if len(value) != 2:
            raise PydanticCustomError("", _errmsg)
        for v in value:
            if type(v) != int:
                raise PydanticCustomError("", _errmsg)
        return value

    @field_validator("size", mode="after")
    @classmethod
    def validate_size_after(cls, value: List[int]) -> Tuple[str, str]:
        """Validate ``size`` after type validation.

        Parameters
        ----------
        value : List[int]
            The value to be validated.

        Returns
        -------
        Tuple[str, str]
            The validated definition of the proximity shell.
        """
        if value[0] >= value[1]:
            _errmsg = "Input value at index 0 must be smaller than input value at index 1"
            raise PydanticCustomError("", _errmsg)
        return (str(value[0]), str(value[1]))

    @field_validator("vdwtype")
    @classmethod
    def validate_vdwtype(cls, value: str) -> str:
        """Validate ``vdwtype``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated van-der-Waals radius method.
        """
        if value not in VDWTYPE_METHODS_KALLISTO:
            _errmsg = f"Input must be one of {VDWTYPE_METHODS_KALLISTO}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateMendeleev(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Mendeleev features.

    For further details, please refer to the Mendeleev documentation
    (https://mendeleev.readthedocs.io/en/stable/, last accessed on 05.09.2025).

    Attributes
    ----------
    method : StrictStr
        The method to use for the effective nuclear charge calculation.
    alle : StrictBool
        Whether to include all valence electrons in the effective nuclear charge calculation.
    """

    method: StrictStr
    alle: StrictBool

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        """Validate ``method``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated method.
        """
        if value not in METHOD_METHODS_MENDELEEV:
            _errmsg = f"Input must be one of {METHOD_METHODS_MENDELEEV}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateMorfeusBuriedVolume(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Morfeus buried volume features.

    For further details, please refer to the Morfeus documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    05.09.2025).

    Attributes
    ----------
    excluded_atoms : List[StrictInt]
        A list of atom indices to be excluded from the feature calculation.
    radii : List[StrictFloat]
        A list of atomic radii to be used for the feature calculation.
    include_hs : StrictBool
        Whether to include hydrogen atoms.
    radius : StrictFloat
        The radius of the reference sphere around the specified atom, must be a positive float.
    radii_type : StrictStr
        The name of the atomic radius scheme to be used for the feature calculation.
    radii_scale : StrictFloat
        A scaling factor for the atomic radii, must be a positive float.
    density : StrictFloat
        The density of the grid points on the molecular surface, must be a positive float.
    z_axis_atoms : List[StrictInt]
        A list of atom indices defining the z-axis.
    xz_plane_atoms : List[StrictInt]
        A list of atom indices defining the xz-plane.
    distal_volume_method : StrictStr
        The method to be used for the distal volume calculation.
    distal_volume_sasa_density : StrictFloat
        The density of the grid points for the distal volume solvent-accessible surface area
        calculation, must be a positive float.
    """

    excluded_atoms: List[StrictInt]
    radii: List[StrictFloat]
    include_hs: StrictBool
    radius: StrictFloat = Field(gt=0)
    radii_type: StrictStr
    radii_scale: StrictFloat = Field(gt=0)
    density: StrictFloat = Field(gt=0)
    z_axis_atoms: List[StrictInt]
    xz_plane_atoms: List[StrictInt]
    distal_volume_method: StrictStr
    distal_volume_sasa_density: StrictFloat = Field(gt=0)

    @field_validator("radii_type")
    @classmethod
    def validate_radii_type(cls, value: str) -> str:
        """Validate ``radii_type``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated radius type.
        """
        if value not in RADII_TYPE_METHODS_MORFEUS_BV_CONE_SOLID_ANGLE:
            _errmsg = f"Input must be one of {RADII_TYPE_METHODS_MORFEUS_BV_CONE_SOLID_ANGLE}"
            raise PydanticCustomError("", _errmsg)
        return value

    @field_validator("distal_volume_method")
    @classmethod
    def validate_distal_volume_method(cls, value: str) -> str:
        """Validate ``distal_volume_method``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated distal volume method.
        """
        if value not in DISTAL_VOLUME_METHODS_MORFEUS_BV:
            _errmsg = f"Input must be one of {DISTAL_VOLUME_METHODS_MORFEUS_BV}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateMorfeusConeAndSolidAngle(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Morfeus cone and solid angle features.

    For further details, please refer to the Morfeus documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    05.09.2025).

    Attributes
    ----------
    radii : List[StrictFloat]
        A list of atomic radii to be used for the feature calculation.
    radii_type : StrictStr
        The name of the atomic radius scheme to be used for the feature calculation.
    density : StrictFloat
        The density of the grid points on the molecular surface, must be a positive float. Only
        relevant for the solid angle calculation.
    """

    radii: List[StrictFloat]
    radii_type: StrictStr
    density: StrictFloat = Field(gt=0)

    @field_validator("radii_type")
    @classmethod
    def validate_radii_type(cls, value: str) -> str:
        """Validate ``radii_type``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated radius type.
        """
        if value not in RADII_TYPE_METHODS_MORFEUS_BV_CONE_SOLID_ANGLE:
            _errmsg = f"Input must be one of {RADII_TYPE_METHODS_MORFEUS_BV_CONE_SOLID_ANGLE}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateMorfeusDispersion(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Morfeus dispersion features.

    For further details, please refer to the Morfeus documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    05.09.2025).

    Attributes
    ----------
    radii : List[StrictFloat]
        A list of atomic radii to be used for the feature calculation.
    radii_type : StrictStr
        The name of the atomic radius scheme to be used for the feature calculation.
    density : StrictFloat
        The density of the grid points on the molecular surface, must be a positive float.
    excluded_atoms : List[StrictInt]
        A list of atom indices to be excluded from the feature calculation.
    included_atoms : List[StrictInt]
        A list of atom indices to be included in the feature calculation.
    """

    radii: List[StrictFloat]
    radii_type: StrictStr
    density: StrictFloat = Field(gt=0)
    excluded_atoms: List[StrictInt]
    included_atoms: List[StrictInt]

    @field_validator("radii_type")
    @classmethod
    def validate_radii_type(cls, value: str) -> str:
        """Validate ``radii_type``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated radius type.
        """
        if value not in RADII_TYPE_METHODS_MORFEUS_DISPERSION:
            _errmsg = f"Input must be one of {RADII_TYPE_METHODS_MORFEUS_DISPERSION}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateMorfeusLocalForce(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Morfeus local force features.

    For further details, please refer to the Morfeus documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    05.09.2025).

    Attributes
    ----------
    method
    project_imag
    imag_cutoff
    save_hessian
    """

    method: StrictStr
    project_imag: StrictBool
    imag_cutoff: StrictFloat = Field(gt=0)
    save_hessian: StrictBool

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        """Validate ``method``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated method.
        """
        if value not in METHODS_MORFEUS_LOCAL_FORCE:
            _errmsg = f"Input must be one of {METHODS_MORFEUS_LOCAL_FORCE}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateMorfeusPyramidalization(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Morfeus pyramidalization features.

    For further details, please refer to the Morfeus documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    05.09.2025).

    Attributes
    ----------
    radii : List[StrictFloat]
        A list of atomic radii to be used for the feature calculation.
    excluded_atoms : List[StrictInt]
        A list of atom indices to be excluded from the feature calculation.
    method : StrictStr
        The name of the pyramidalization calculation method.
    scale_factor : StrictFloat
        A scaling factor for determining connectivity.
    """

    radii: List[StrictFloat]
    excluded_atoms: List[StrictInt]
    method: StrictStr
    scale_factor: StrictFloat = Field(gt=0)

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        """Validate ``method``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated method to calculate the pyramidalization.
        """
        if value not in PYRAMIDALIZATION_CALCULATION_METHODS_MORFEUS_PYRAMIDALIZATION:
            _errmsg = f"Input must be one of {PYRAMIDALIZATION_CALCULATION_METHODS_MORFEUS_PYRAMIDALIZATION}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateMorfeusSasa(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Morfeus solvent-accessible surface area
    features.

    For further details, please refer to the Morfeus documentation
    (https://digital-chemistry-laboratory.github.io/morfeus/index.html, last accessed on
    05.09.2025).

    Attributes
    ----------
    radii : List[StrictFloat]
        A list of atomic radii to be used for the SASA calculation.
    radii_type : StrictStr
        The name of the atomic radius scheme to be used for the SASA calculation.
    probe_radius : StrictFloat
        The radius of the probe sphere, must be a positive float.
    density : StrictFloat
        The density of the grid points on the molecular surface, must be a positive float.
    """

    radii: List[StrictFloat]
    radii_type: StrictStr
    probe_radius: StrictFloat = Field(gt=0)
    density: StrictFloat = Field(gt=0)

    @field_validator("radii_type")
    @classmethod
    def validate_radii_type(cls, value: str) -> str:
        """Validate ``radii_type``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the formatted and validated radius type.
        """
        if value not in RADII_TYPE_METHODS_MORFEUS_SASA:
            _errmsg = f"Input must be one of {RADII_TYPE_METHODS_MORFEUS_SASA}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateMultiwfnRootData(BaseModel):
    """Validate the configuration settings for Multiwfn's root data.

    For further details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last
    accessed on 05.09.2025).

    Attributes
    ----------
    OMP_STACKSIZE : StrictStr
        The size of the OpenMP stack.
    NUM_THREADS : StrictInt
        The number of threads, must be a positive integer.
    """

    OMP_STACKSIZE: Optional[StrictStr] = Field(default=None)
    NUM_THREADS: Optional[StrictInt] = Field(default=None, gt=0)


class ValidateMultiwfnBondAnalysis(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Multiwfn bond analysis features.

    For further details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last
    accessed on 05.09.2025).

    Attributes
    ----------
    OMP_STACKSIZE : StrictStr
        The size of the OpenMP stack.
    NUM_THREADS : StrictInt
        The number of threads, must be a positive integer.
    ibsi_grid : StrictStr
        The quality of the grid for the calculation of the intrinsic bond strength index.
    connectivity_index_threshold : StrictFloat
        The threshold for considering atom connectivity, must be a positive float.
    """

    OMP_STACKSIZE: Optional[StrictStr] = Field(default=None)
    NUM_THREADS: Optional[StrictInt] = Field(default=None, gt=0)

    ibis_igm_type: StrictStr
    ibsi_grid: StrictStr
    connectivity_index_threshold: StrictFloat = Field(gt=0)

    @field_validator("ibis_igm_type")
    @classmethod
    def validate_ibis_igm_type(cls, value: str) -> str:
        """Validate ``ibis_igm_type``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The name of the selected IGM type
        """
        if value not in IGM_TYPES_MULTIWFN_BOND_ANALYSIS:
            _errmsg = f"Input must be one of {IGM_TYPES_MULTIWFN_BOND_ANALYSIS}"
            raise PydanticCustomError("", _errmsg)
        return value

    @field_validator("ibsi_grid")
    @classmethod
    def validate_ibsi_grid(cls, value: Any) -> int:
        """Validate ``ibsi_grid``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        int
            The index of the selected grid quality.
        """
        _keys = list(IBIS_GRID_METHODS_MULTIWFN_BOND_ANALYSIS.keys())
        if value not in _keys:
            _errmsg = f"Input must be one of {_keys}"
            raise PydanticCustomError("", _errmsg)
        return IBIS_GRID_METHODS_MULTIWFN_BOND_ANALYSIS[value]


class ValidateMultiwfnCdft(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Multiwfn conceptual DFT features.

    For further details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last
    accessed on 05.09.2025).

    Attributes
    ----------
    OMP_STACKSIZE : StrictStr
        The size of the OpenMP stack.
    NUM_THREADS : StrictInt
        The number of threads, must be a positive integer.
    iterable_option : List[StrictStr]
        A list of population analysis schemes to be used for the calculation of the conceptual DFT
        features.
    ow_delta : StrictFloat
        The delta parameter for the calculation of orbital-weighted Fukui indices, must be a
        positive float.
    """

    OMP_STACKSIZE: Optional[StrictStr] = Field(default=None)
    NUM_THREADS: Optional[StrictInt] = Field(default=None, gt=0)

    iterable_option: List[StrictStr]
    ow_delta: StrictFloat = Field(gt=0)

    @field_validator("iterable_option", mode="before")
    @classmethod
    def validate_iterable_option_before(cls, value: Any) -> Any:
        """Validate ``iterable_option`` before type validation.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        Any
            The pre-validated iterable options.
        """
        if any([value is None, value == "", value == []]):
            _errmsg = "Input must not be empty"
            raise PydanticCustomError("", _errmsg)
        elif type(value) == str:
            value = [value.strip().lower()]
        elif type(value) == list:
            try:
                value = [str(v).strip().lower() for v in value]
            except:
                pass
        return value

    @field_validator("iterable_option", mode="after")
    @classmethod
    def validate_iterable_option_after(cls, value: List[str]) -> List[str]:
        """Validate ``iterable_option`` after type validation.

        Parameters
        ----------
        value : List[str]
            The value to be validated.

        Returns
        -------
        List[str]
            The validated iterable.
        """
        for val in value:
            if val not in ITERABLE_OPTIONS_MULTIWFN_CDFT:
                _errmsg = f"Input must only contain {ITERABLE_OPTIONS_MULTIWFN_CDFT}"
                raise PydanticCustomError("", _errmsg)
        return value


class ValidateMultiwfnFuzzy(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Multiwfn fuzzy space analysis features.

    For further details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last
    accessed on 05.09.2025).

    Attributes
    ----------
    OMP_STACKSIZE : StrictStr
        The size of the OpenMP stack.
    NUM_THREADS : StrictInt
        The number of threads, must be a positive integer.
    integration_grid : StrictStr
        The name of the integration grid method.
    exclude_atoms : List[StrictInt]
        A list of atom indices to be excluded from the feature calculation.
    n_iterations_becke_partition : StrictInt
        The number of iterations for the Becke partitioning, must be a positive integer.
    radius_becke_partition : StrictStr
        The name of the method for the radius in Becke partitioning.
    partitioning_scheme : StrictStr
        The name of the partitioning scheme.
    real_space_function : StrictStr
        The name of the real space function to be used.
    """

    OMP_STACKSIZE: Optional[StrictStr] = Field(default=None)
    NUM_THREADS: Optional[StrictInt] = Field(default=None, gt=0)

    integration_grid: StrictStr
    exclude_atoms: List[StrictInt]
    n_iterations_becke_partition: StrictInt = Field(gt=0)
    radius_becke_partition: StrictStr
    partitioning_scheme: StrictStr
    real_space_function: StrictStr

    @field_validator("integration_grid")
    @classmethod
    def validate_integration_grid(cls, value: Any) -> int:
        """Validate ``integration_grid``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        int
            The index of the selected integration grid method.
        """
        _keys = list(INTEGRATION_GRID_METHODS_MULTIWFN_FUZZY.keys())
        if value not in _keys:
            _errmsg = f"Input must be one of {_keys}"
            raise PydanticCustomError("", _errmsg)
        return INTEGRATION_GRID_METHODS_MULTIWFN_FUZZY[value]

    @field_validator("radius_becke_partition")
    @classmethod
    def validate_radius_becke_partition(cls, value: Any) -> int:
        """Validate ``radius_becke_partition``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        int
            The index of the selected radius method for Becke partitioning.
        """
        _keys = list(RADIUS_BECKE_PARTITION_METHODS_MULTIWFN_FUZZY.keys())
        if value not in _keys:
            _errmsg = f"Input must be one of {_keys}"
            raise PydanticCustomError("", _errmsg)
        return RADIUS_BECKE_PARTITION_METHODS_MULTIWFN_FUZZY[value]

    @field_validator("partitioning_scheme")
    @classmethod
    def validate_partitioning_scheme(cls, value: Any) -> int:
        """Validate ``partitioning_scheme``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        int
            The index of the selected partitioning scheme.
        """
        _keys = list(PARTITION_SCHEME_METHODS_MULTIWFN_FUZZY.keys())
        if value not in _keys:
            _errmsg = f"Input must be one of {_keys}"
            raise PydanticCustomError("", _errmsg)
        return PARTITION_SCHEME_METHODS_MULTIWFN_FUZZY[value]

    @field_validator("real_space_function")
    @classmethod
    def validate_real_space_function(cls, value: Any) -> int:
        """Validate ``real_space_function``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        int
            The index of the selected real space function.
        """
        _keys = list(REAL_SPACE_FUNCTIONS_MULTIWFN.keys())
        if value not in _keys:
            _errmsg = f"Input must be one of {_keys}"
            raise PydanticCustomError("", _errmsg)
        return REAL_SPACE_FUNCTIONS_MULTIWFN[value]


class ValidateMultiwfnMisc(_StandardizeStrMixin, BaseModel):
    """Validate the miscellaneous configuration settings for the Multiwfn features.

    For further details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last
    accessed on 05.09.2025).

    Attributes
    ----------
    OMP_STACKSIZE : StrictStr
        The size of the OpenMP stack.
    NUM_THREADS : StrictInt
        The number of threads, must be a positive integer.
    """

    OMP_STACKSIZE: Optional[StrictStr] = Field(default=None)
    NUM_THREADS: Optional[StrictInt] = Field(default=None, gt=0)


class ValidateMultiwfnOrbital(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Multiwfn orbital features.

    For further details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last
    accessed on 05.09.2025).

    Attributes
    ----------
    OMP_STACKSIZE : StrictStr
        The size of the OpenMP stack.
    NUM_THREADS : StrictInt
        The number of threads, must be a positive integer.
    homo_minus : StrictInt
        The number of orbitals to go below the HOMO, must be great than or equal to zero.
    lumo_plus : StrictInt
        The number of orbitals to go above the LUMO, must be great than or equal to zero.
    """

    OMP_STACKSIZE: Optional[StrictStr] = Field(default=None)
    NUM_THREADS: Optional[StrictInt] = Field(default=None, gt=0)

    homo_minus: StrictInt = Field(ge=0)
    lumo_plus: StrictInt = Field(ge=0)


class ValidateMultiwfnPopulation(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Multiwfn population analysis features.

    For further details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last
    accessed on 05.09.2025).

    Attributes
    ----------
    OMP_STACKSIZE : StrictStr
        The size of the OpenMP stack.
    NUM_THREADS : StrictInt
        The number of threads, must be a positive integer.
    n_iterations_becke_partition : StrictInt
        The number of iterations for the Becke partitioning, must be a positive integer.
    radius_becke_partition : StrictStr
        The name of the method for the radius in Becke partitioning.
    grid_spacing_chelpg : StrictFloat
        The grid size for CHELPG calculations.
    box_extension_chelpg : StrictFloat
        The box extension size for CHELPG calculations.
    esp_type : StrictStr
        The name of the ESP type for various population analysis methods.
    atomic_radii : StrictStr
        The name of the atomic radii definition used in various population analysis methods.
    exclude_atoms : List[StrictInt]
        A list of atom indices to be excluded from the feature calculation.
    fitting_points_settings_merz_kollmann : List[StrictFloat]
        A list with the number and the scale factors required for calculating the Merz-Kollmann
        fitting points.
    n_points_angstrom2_merz_kollmann : StrictFloat
        The number of fitting points per square Angstrom for Merz-Kollmann fitting.
    eem_parameters : StrictStr
        The name of the parameter set for calculating EEM charges.
    tightness_resp : StrictFloat
        The tightness parameter for RESP calculations.
    restraint_one_stage_resp : StrictFloat
        The restraint strength for one-stage RESP calculations.
    restraint_stage1_resp : StrictFloat
        The restraint strength for stage 1 of two-stage RESP calculations.
    restraint_stage2_resp : StrictFloat
        The restraint strength for stage 2 of two-stage RESP calculations.
    n_iterations_resp : StrictInt
        The maximum number of iterations for RESP calculations.
    convergence_threshold_resp : StrictFloat
        The convergence threshold for RESP calculations.
    ch_equivalence_constraint_resp : StrictBool
        Whether to apply charge equivalence constraints due to chemical equivalence in RESP
        calculation.
    """

    OMP_STACKSIZE: Optional[StrictStr] = Field(default=None)
    NUM_THREADS: Optional[StrictInt] = Field(default=None, gt=0)

    n_iterations_becke_partition: StrictInt = Field(gt=0)
    radius_becke_partition: StrictStr
    grid_spacing_chelpg: StrictFloat = Field(gt=0)
    box_extension_chelpg: StrictFloat = Field(gt=0)
    esp_type: StrictStr
    atomic_radii: StrictStr
    exclude_atoms: List[StrictInt]
    fitting_points_settings_merz_kollmann: List[StrictFloat]
    n_points_angstrom2_merz_kollmann: StrictFloat = Field(gt=0)
    eem_parameters: StrictStr
    tightness_resp: StrictFloat = Field(gt=0)
    restraint_one_stage_resp: StrictFloat = Field(gt=0)
    restraint_stage1_resp: StrictFloat = Field(gt=0)
    restraint_stage2_resp: StrictFloat = Field(gt=0)
    n_iterations_resp: StrictInt = Field(gt=0)
    convergence_threshold_resp: StrictFloat = Field(gt=0)
    ch_equivalence_constraint_resp: StrictBool

    @field_validator("radius_becke_partition")
    @classmethod
    def validate_radius_becke_partition(cls, value: Any) -> int:
        """Validate ``radius_becke_partition``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        int
            The index of the selected radius method for Becke partitioning.
        """
        _keys = list(RADIUS_BECKE_PARTITION_METHODS_MULTIWFN_POPULATION.keys())
        if value not in _keys:
            _errmsg = f"Input must be one of {_keys}"
            raise PydanticCustomError("", _errmsg)
        return RADIUS_BECKE_PARTITION_METHODS_MULTIWFN_POPULATION[value]

    @field_validator("esp_type")
    @classmethod
    def validate_esp_type(cls, value: Any) -> int:
        """Validate ``esp_type``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        int
            The index of the selected ESP type.
        """
        _keys = list(ESP_TYPE_MULTIWFN_POPULATION.keys())
        if value not in _keys:
            _errmsg = f"Input must be one of {_keys}"
            raise PydanticCustomError("", _errmsg)
        return ESP_TYPE_MULTIWFN_POPULATION[value]

    @field_validator("atomic_radii")
    @classmethod
    def validate_atomic_radii(cls, value: Any) -> int:
        """Validate ``atomic_radii``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        int
            The index of the radius type.
        """
        _keys = list(ATOMIC_RADII_MULTIWFN_POPULATION.keys())
        if value not in _keys:
            _errmsg = f"Input must be one of {_keys}"
            raise PydanticCustomError("", _errmsg)
        return ATOMIC_RADII_MULTIWFN_POPULATION[value]

    @field_validator("eem_parameters")
    @classmethod
    def validate_eem_parameters(cls, value: Any) -> int:
        """Validate ``eem_parameters``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        int
            The index of the EEM parameter set.
        """
        _keys = list(EEM_PARAMETERS_MULTIWFN_POPULATION.keys())
        if value not in _keys:
            _errmsg = f"Input must be one of {_keys}"
            raise PydanticCustomError("", _errmsg)
        return EEM_PARAMETERS_MULTIWFN_POPULATION[value]

    @field_validator("fitting_points_settings_merz_kollmann")
    @classmethod
    def validate_fitting_points_settings_merz_kollmann(cls, value: Any) -> List[float]:
        """Validate ``fitting_points_settings_merz_kollmann``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        List[float]
            The validated number and scale factors of the layers of MK fitting points.
        """
        for v in value:
            if v <= 0:
                _errmsg = "All input values must be greater than 0"
                raise PydanticCustomError("", _errmsg)
        return [float(v) for v in value]


class ValidateMultiwfnSurface(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Multiwfn surface features.

    For further details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last
    accessed on 05.09.2025).

    Attributes
    ----------
    OMP_STACKSIZE : StrictStr
        The size of the OpenMP stack.
    NUM_THREADS : StrictInt
        The number of threads, must be a positive integer.
    surface_definition : StrictStr
        The scheme to define the molecular surface.
    surface_iso_value : StrictFloat
        The iso value for defining the surface, must be a positive float.
    grid_point_spacing : StrictFloat
        The scaling parameter for the grid to generate the surface, must be a positive float.
    length_scale : StrictFloat
        The length scale for surface generation, must be a positive float
    orbital_overlap_edr_option : List[Any]
        The total number, start, and increment in EDR exponents.
    """

    OMP_STACKSIZE: Optional[StrictStr] = Field(default=None)
    NUM_THREADS: Optional[StrictInt] = Field(default=None, gt=0)

    surface_definition: StrictStr
    surface_iso_value: StrictFloat = Field(gt=0)
    grid_point_spacing: StrictFloat = Field(gt=0)
    length_scale: StrictFloat = Field(gt=0)
    orbital_overlap_edr_option: List[Any]

    @field_validator("surface_definition")
    @classmethod
    def validate_surface_definition(cls, value: Any) -> int:
        """Validate ``surface_definition``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        int
            The index of the selected surface definition.
        """
        _keys = list(REAL_SPACE_FUNCTIONS_MULTIWFN.keys())
        if value not in _keys:
            _errmsg = f"Input must be one of {_keys}"
            raise PydanticCustomError("", _errmsg)
        return REAL_SPACE_FUNCTIONS_MULTIWFN[value]

    @field_validator("orbital_overlap_edr_option")
    @classmethod
    def validate_orbital_overlap_edr_option(cls, value: List[Any]) -> List[Union[int, float]]:
        """Validate ``orbital_overlap_edr_option``.

        Parameters
        ----------
        value : List[Any]
            The value to be validated.

        Returns
        -------
        List[Union[int, float]]
            The validated list of the EDR function data.
        """
        if len(value) != 3:
            _errmsg = "Input must exactly contain 3 values"
            raise PydanticCustomError("", _errmsg)

        if type(value[0]) != int:
            _errmsg = "Input must contain an integer value at index 0"
            raise PydanticCustomError("", _errmsg)
        if value[0] < 1:
            _errmsg = "Input at index 0 must be greater than 0"
            raise PydanticCustomError("", _errmsg)
        if value[0] > 50:
            _errmsg = "Input at index 0 must not be greater than 50"
            raise PydanticCustomError("", _errmsg)

        if type(value[1]) != int and type(value[1]) != float:
            _errmsg = "Input must contain a number at index 1"
            raise PydanticCustomError("", _errmsg)

        if type(value[2]) != int and type(value[2]) != float:
            _errmsg = "Input must contain a number at index 2"
            raise PydanticCustomError("", _errmsg)
        if value[2] <= 1.01:
            _errmsg = "Input at index 2 must be greater than 1.01"
            raise PydanticCustomError("", _errmsg)

        return value


class ValidateMultiwfnTopology(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for the Multiwfn topology features.

    For further details, please refer to the Multiwfn manual (http://sobereva.com/multiwfn/, last
    accessed on 05.09.2025).

    Attributes
    ----------
    OMP_STACKSIZE : StrictStr
        The size of the OpenMP stack.
    NUM_THREADS : StrictInt
        The number of threads, must be a positive integer.
    step_size : StrictFloat
        The step size, must be a positive float.
    neighbor_distance_cutoff : StrictFloat
        The neighbor distance cutoff, must be a positive float.
    """

    OMP_STACKSIZE: Optional[StrictStr] = Field(default=None)
    NUM_THREADS: Optional[StrictInt] = Field(default=None, gt=0)

    step_size: StrictFloat = Field(gt=0)
    neighbor_distance_cutoff: StrictFloat = Field(gt=0)


class ValidatePsi4(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for Psi4.

    For further details, please refer to the Psi4 documentation
    (https://psicode.org/psi4manual/master/index.html, last accessed on 05.09.2025).

    Attributes
    ----------
    basis : str
        The basis set.
    CLEAN_SCRATCH_AFTER_CALCULATION : StrictBool
        Whether to clean the scratch directory after the calculation.
    method : StrictStr
        The quantum chemistry method.
    memory : str
        The amount of memory, e.g., "2 gb".
    maxiter : int
        The maximum number of SCF iterations.
    num_threads : int
        The number of threads.
    PSI_SCRATCH : StrictStr
        The path to the scratch base directory for Psi4 calculations.
    solvent : str
        The name of the solvent.
    solvent_model_solver : str
        The name of the solver for the solvent model.
    """

    PSI_SCRATCH: StrictStr = Field(default="/tmp/")
    CLEAN_SCRATCH_AFTER_CALCULATION: StrictBool = Field(default=True)
    method: StrictStr
    basis: StrictStr
    maxiter: StrictInt = Field(gt=0)
    memory: StrictStr
    num_threads: StrictInt = Field(gt=0)
    solvent: StrictStr
    solvent_model_solver: StrictStr

    @field_validator("memory")
    @classmethod
    def validate_memory(cls, value: str) -> str:
        """Validate ``memory``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The validated memory string.
        """
        _errmsg = "Input must be a string in the format '<number> <unit>', e.g., '2 gb'"
        splitted = value.split()
        if len(splitted) != 2:
            raise PydanticCustomError("", _errmsg)
        try:
            int(splitted[0])
        except ValueError:
            raise PydanticCustomError("", _errmsg)

        return value.strip()

    @field_validator("solvent")
    @classmethod
    def validate_solvent(cls, value: str) -> str:
        """Validate ``solvent``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The validated solvent string.
        """
        if value not in SOLVENTS_PSI4:
            _errmsg = f"Input must be one of {SOLVENTS_PSI4}"
            raise PydanticCustomError("", _errmsg)
        return value

    @field_validator("solvent_model_solver")
    @classmethod
    def validate_solvent_model_solver(cls, value: str) -> str:
        """Validate ``solvent_model_solver``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The validated solver string string.
        """
        if value not in SOLVENT_MODEL_SOLVERS_PSI4:
            _errmsg = f"Input must be one of {SOLVENT_MODEL_SOLVERS_PSI4}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateRdkitFingerprint(BaseModel):
    """Validate the configuration settings for the RDKit fingerprint features.

    For further details, please refer to the RDKit documentation
    (https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html, last accessed on
    05.09.2025).

    Attributes
    ----------
    radius : StrictInt
        The radius of the fingerprint, must be a non-negative integer.
    countSimulation : StrictBool
        Whether to use count simulation during fingerprint generation.
    includeChirality : StrictBool
        Whether to include chirality information in the fingerprint.
    useBondTypes : StrictBool
        Whether to consider bond types in the fingerprint.
    countBounds : Any
        The boundaries for count simulation.
    fpSize : StrictInt
        The size of the fingerprint, must be a positive integer.
    torsionAtomCount : StrictInt
        The number of atoms to include in the torsions.
    minDistance : StrictInt
        The minimum distance between two atoms, must be a non-negative integer.
    maxDistance : StrictInt
        The maximum distance between two atoms, must be a non-negative integer.
    use2D : StrictBool
        Whether to use the 2D distance matrix during fingerprint generation.
    minPath : StrictInt
        The minimum path length as number of bonds, must be a non-negative integer.
    maxPath : StrictInt
        The maximum path length as number of bonds, must be a non-negative integer.
    useHs : StrictBool
        Whether to include hydrogen atoms in the fingerprint.
    branchedPaths : StrictBool
        Whether to consider branched paths in the fingerprint.
    useBondOrder : StrictBool
        Whether to consider bond order in the fingerprint.
    numBitsPerFeature : StrictInt
        The number of bits to use per feature, must be a positive integer.
    """

    radius: StrictInt = Field(ge=0)
    countSimulation: StrictBool
    includeChirality: StrictBool
    useBondTypes: StrictBool
    countBounds: Any
    fpSize: StrictInt = Field(gt=0)
    torsionAtomCount: StrictInt = Field(ge=0)
    minDistance: StrictInt = Field(ge=0)
    maxDistance: StrictInt = Field(ge=0)
    use2D: StrictBool
    minPath: StrictInt = Field(ge=0)
    maxPath: StrictInt = Field(ge=0)
    useHs: StrictBool
    branchedPaths: StrictBool
    useBondOrder: StrictBool
    numBitsPerFeature: StrictInt = Field(gt=0)

    @field_validator("countBounds")
    @classmethod
    def validate_count_bounds(cls, value: Any) -> Any:
        """Validate ``countBounds``.

        Parameters
        ----------
        value : Any
            The value to be validated.

        Returns
        -------
        Any
            The validated value, either ``None`` or the original value specified by the user.
        """
        if type(value) == str:
            if value.strip().lower() == "none":
                return None
        return value


class ValidateXtb(_StandardizeStrMixin, BaseModel):
    """Validate the configuration settings for xtb.

    For further details, please refer to the xtb documentation
    (https://xtb-docs.readthedocs.io/en/latest/, last accessed on 05.09.2025).

    Attributes
    ----------
    OMP_STACKSIZE : StrictStr
        The size of the OpenMP stack.
    OMP_NUM_THREADS : StrictInt
        The number of OpenMP threads, must be a positive integer.
    OMP_MAX_ACTIVE_LEVELS : StrictInt
        The maximum number of nested active parallel regions, must be a positive integer.
    MKL_NUM_THREADS : StrictInt
        The number of threads for the Intel Math Kernel Library, must be a positive integer.
    XTBHOME : StrictStr
        The path to the xtb home directory. If set to "auto", the path is determined automatically.
    method : StrictStr
        The semi-empirical method to be used.
    iterations : StrictInt
        The maximum number of SCF iterations, must be a positive integer.
    acc : StrictFloat
        The accuracy level for the xtb calculation.
    etemp : StrictInt
        The electronic temperature.
    etemp_native : StrictInt
        The electronic temperature used for the direct calculation xtb features.
    solvent_model : str
        The name of the solvent model.
    solvent : str
        The name of the solvent.
    """

    OMP_STACKSIZE: Optional[StrictStr] = Field(default=None)
    OMP_NUM_THREADS: Optional[StrictInt] = Field(default=None, gt=0)
    OMP_MAX_ACTIVE_LEVELS: Optional[StrictInt] = Field(default=None, gt=0)
    MKL_NUM_THREADS: Optional[StrictInt] = Field(default=None, gt=0)
    XTBHOME: Optional[StrictStr] = Field(default=None)

    method: StrictStr
    iterations: StrictInt = Field(gt=0)
    acc: StrictFloat = Field(ge=0.0001, le=1000)
    etemp: StrictInt = Field(ge=0)
    etemp_native: StrictInt = Field(ge=0)
    solvent_model: StrictStr
    solvent: StrictStr

    @field_validator("XTBHOME")
    @classmethod
    def validate_xtb_home(cls, value: Optional[str]) -> Optional[str]:
        """Validate ``XTBHOME``.

        If set to "auto", the path is determined automatically by pointing to /share/xtb
        in the xtb installation directory. If the user-provided path does not exist, the
        automatically generated path is used. If set to ``None``, ``None`` is returned.

        Parameters
        ----------
        value : Optional[str]
            The value to be validated.

        Returns
        -------
        Optional[str]
            The validated XTB home path, either the user-provided path, the automatically
            generated one, or ``None``.
        """
        if value is None:
            return value

        _val = value.lower()

        if _val != "auto" and os.path.exists(value) is True:
            return value

        _xtb_path = shutil.which(cmd="xtb")
        if _xtb_path is None:
            if _val == "auto":
                _errmsg = "XTBHOME set to 'auto' but xtb executable not found on PATH"
            else:
                _errmsg = (
                    f"XTBHOME path '{value}' does not exist and xtb executable was not "
                    "found on PATH to derive it automatically"
                )
            raise PydanticCustomError("", _errmsg)

        _auto_value = os.path.join(os.path.dirname(os.path.dirname(_xtb_path)), "share", "xtb")

        if os.path.exists(_auto_value) is False:
            _errmsg = f"Auto-derived XTBHOME '{_auto_value}' does not exist"
            raise PydanticCustomError("", _errmsg)

        return _auto_value

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        """Validate ``method``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The formatted and validated method string.
        """
        if value not in METHODS_XTB:
            _errmsg = f"Input must be one of {METHODS_XTB}"
            raise PydanticCustomError("", _errmsg)
        return value

    @field_validator("solvent_model")
    @classmethod
    def validate_solvent_model(cls, value: str) -> str:
        """Validate ``solvent_model``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The formatted and validated solvent model string.
        """
        if value not in SOLVENT_MODELS_XTB:
            _errmsg = f"Input must be one of {SOLVENT_MODELS_XTB}"
            raise PydanticCustomError("", _errmsg)
        return value

    @field_validator("solvent")
    @classmethod
    def validate_solvent(cls, value: str) -> str:
        """Validate ``solvent``.

        Parameters
        ----------
        value : str
            The value to be validated.

        Returns
        -------
        str
            The formatted and validated solvent string.
        """
        if value not in SOLVENTS_XTB:
            _errmsg = f"Input must be one of {SOLVENTS_XTB}"
            raise PydanticCustomError("", _errmsg)
        return value


class ValidateDummy(BaseModel):
    """Dummy validator class that does not perform any validation."""

    pass


def config_data_validator(
    config_path: List[str], params: Dict[str, Any], _namespace: Optional[str]
) -> Dict[str, Any]:
    """Validate the configuration settings of a featurizer.

    The respective validation class is selected based on the provided configuration path. In case
    no validation is needed or implemented, a warning is logged and a dummy validator is called.

    Parameters
    ----------
    config_path : List[str]
        A list of strings representing the path to the configuration settings in the internal
        configuration settings tree.
    params : Dict[str, Any]
        A dictionary containing the configuration settings to be validated. The keys should match
        the attributes of the respective validation data class.
    _namespace : Optional[str]
        The namespace of the currently handled molecule for logging purposes; ``None`` if no
        molecule was read in yet.

    Returns
    -------
    Dict[str, Any]
        The validated and formatted configuration settings.
    """
    _loc = get_function_or_method_name()

    _validators = {
        "alfabet": ValidateAlfabet,
        "bonafide.autocorrelation": ValidateBonafideAutocorrelation,
        "bonafide.constant": ValidateBonafideConstant,
        "bonafide.distance": ValidateBonafideDistance,
        "bonafide.functional_group": ValidateBonafideFunctionalGroup,
        "bonafide.misc": ValidateDummy,
        "bonafide.oxidation_state": ValidateBonafideOxidationState,
        "bonafide.symmetry": ValidateBonafideSymmetry,
        "dbstep": ValidateDbstep,
        "dscribe.acsf": ValidateDscribeAcsf,
        "dscribe.coulomb_matrix": ValidateDscribeCoulombMatrix,
        "dscribe.lmbtr": ValidateDscribeLmbtr,
        "dscribe.soap": ValidateDscribeSoap,
        "kallisto": ValidateKallisto,
        "mendeleev": ValidateMendeleev,
        "morfeus.buried_volume": ValidateMorfeusBuriedVolume,
        "morfeus.cone_and_solid_angle": ValidateMorfeusConeAndSolidAngle,
        "morfeus.dispersion": ValidateMorfeusDispersion,
        "morfeus.local_force": ValidateMorfeusLocalForce,
        "morfeus.pyramidalization": ValidateMorfeusPyramidalization,
        "morfeus.sasa": ValidateMorfeusSasa,
        "multiwfn": ValidateMultiwfnRootData,
        "multiwfn.bond_analysis": ValidateMultiwfnBondAnalysis,
        "multiwfn.cdft": ValidateMultiwfnCdft,
        "multiwfn.fuzzy": ValidateMultiwfnFuzzy,
        "multiwfn.misc": ValidateMultiwfnMisc,
        "multiwfn.orbital": ValidateMultiwfnOrbital,
        "multiwfn.population": ValidateMultiwfnPopulation,
        "multiwfn.surface": ValidateMultiwfnSurface,
        "multiwfn.topology": ValidateMultiwfnTopology,
        "psi4": ValidatePsi4,
        "qmdesc": ValidateDummy,
        "rdkit.fingerprint": ValidateRdkitFingerprint,
        "rdkit.misc": ValidateDummy,
        "xtb": ValidateXtb,
    }
    config_path_str = ".".join(config_path)

    logging.info(
        f"'{_namespace}' | {_loc}()\nValidating configuration settings from '{config_path_str}'."
    )

    # In case no validator is implemented
    if config_path_str not in _validators:
        logging.warning(
            f"'{_namespace}' | {_loc}()\nNo configuration settings validation class implemented "
            f"for '{config_path_str}'. This is probably due to using a custom featurization "
            "method. Ensure that its setting have the correct data type and format. No data "
            "validation is performed."
        )
        params = {key: value for key, value in params.items() if key not in ["feature_info"]}
        logging.info(f"'{_namespace}' | {_loc}()\nConfiguration settings: {params}.")
        return params

    # Try to validate the set of parameters
    try:
        v = _validators[config_path_str](**params)
    except ValidationError as e:
        error_dict = defaultdict(list)
        for error in e.errors():
            _p_loc = error["loc"]
            _inp = error["input"]

            # check_iterable_option (mode=after) raises errors with empty loc tuple as it is not a
            # classmethod. This is only the case for one method, which checks iterable options.
            # Therefore, the location is set manually here.
            if len(_p_loc) == 0:
                _p_loc = ("iterable_option",)
                _inp = {key: value for key, value in _inp.items() if key not in ["feature_info"]}
                error_dict[str(_p_loc[0])].append(f"{error['msg']}.")

            else:
                error_dict[str(_p_loc[0])].append(
                    f"{error['msg']}, obtained: {_inp} (of type '{type(error['input']).__name__}')."
                )

        _errmsg = f"Incorrect data encountered in '{config_path_str}': {dict(error_dict)}"
        logging.error(f"'{_namespace}' | {_loc}()\n{_errmsg}")
        raise ValueError(f"{_loc}(): {_errmsg}")

    # Remove the feature_info parameter (is now irrelevant)
    params = {key: value for key, value in v.__dict__.items() if key not in ["feature_info"]}

    logging.info(f"'{_namespace}' | {_loc}()\nValidated configuration settings: {params}.")
    return params
