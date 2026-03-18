"""Base class for all feature factory classes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from bonafide.utils.base_mixin import _BaseMixin
from bonafide.utils.helper_functions import get_function_or_method_name

if TYPE_CHECKING:
    from rdkit import Chem


class BaseFeaturizer(_BaseMixin):
    """Base class for all feature factory classes.

    All feature factory classes must inherit from this class. It provides the basic structure and
    workflow for generating and storing features through its ``__call__()`` method as well as
    additional helper methods for caching feature values.

    Attributes
    ----------
    _err : Optional[str]
        The error message generated during feature calculation, if any. It is returned by the
        ``__call__()`` method. It is ``None`` if no error occurred.
    _out : Optional[Union[int, float, bool, str]]
        The output of the feature calculation (feature value for a given atom or bond of a given
        conformer) that is returned by the ``__call__()`` method. It is ``None`` if an error
        occurred.
    atom_bond_idx : int
        The index of the atom or bond for which the feature is requested.
    conformer_idx : int
        The index of the conformer in the molecule vault.
    conformer_name : str
        The name of the conformer for which the feature is requested.
    extraction_mode : str
        Indicator if the ``calculate()`` method of a respective feature factory calculates the
        features for all atoms or bonds of the molecule when called once ("multi") or only for
        a single atom or bond ("single"). It must be set in the child class.
    feature_cache : List[Dict[str, Dict[int, Optional[Union[str, bool, int, float]]]]]
        The cache of atom or bond features for each conformer. The individual list entries are
        dictionaries with the feature names as keys and dictionaries mapping atom indices to
        feature values as values.
    feature_name : str
        The name of the feature that is requested.
    feature_type : str
        The type of the feature that is requested, either "atom" or "bond".
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object of the conformer for which the feature is requested.
    results : Dict[int, Dict[str, Optional[Union[int, float, bool, str]]]]
        Dictionary for storing the results of the feature calculation. Its keys are the atom or
        bond indices, and the values are dictionaries with the feature name(s) as key(s) and their
        values. It is populated by the ``calculate()`` method implemented in the child classes
        (feature factory).
    """

    atom_bond_idx: int
    conformer_idx: int
    conformer_name: str
    extraction_mode: str
    feature_cache: List[Dict[str, Dict[int, Optional[Union[str, bool, int, float]]]]]
    feature_name: str
    feature_type: str
    mol: Chem.rdchem.Mol

    results: Dict[int, Dict[str, Optional[Union[int, float, bool, str]]]]
    _out: Optional[Union[int, float, bool, str]]
    _err: Optional[str]

    def __init__(self) -> None:
        self.results = {}

        self._out = None
        self._err = None

        # Check if feature factory is correctly implemented
        self._check_requirements()

    def __call__(
        self, **kwargs: Any
    ) -> Tuple[Optional[Union[int, float, bool, str]], Optional[str]]:
        """Calculate the feature value for an atom or bond of a conformer.

        Initially, it is attempted to pull and return the requested data from the feature cache.
        If the data is not available, it is calculated. After that, all the data contained in
        ``results`` is written to the cache, and the requested data is pulled from there and
        returned. Finally, the output files generated during the calculation are saved (if
        requested by the user), and the working directory is deleted.

        If an unexpected error occurs during the feature calculation which is not captured by the
        ``_err`` attribute, it is logged and raised as ``RuntimeError``.

        Parameters
        ----------
        **kwargs: Any
            Optional arguments that are set as attributes of the class instance. This allows
            passing different data to the child classes through the ``__call__()`` method.

        Returns
        -------
        Tuple[Optional[Union[int, float, bool, str]], Optional[str]]
            A tuple containing the feature value (``None`` if an error occurred) and an error
            message (``None`` if no error occurred).
        """
        # Set all attributes required for the feature calculation
        for attr_name, value in kwargs.items():
            setattr(self, attr_name, value)

        _loc = f"{self.__class__.__name__}.calculate"
        _namespace = self.conformer_name[::-1].split("__", 1)[-1][::-1]

        # Try to get the data from the cache (in case it was already calculated)
        self._from_cache()
        if self._out is not None:
            return self._out, self._err

        # Set up a working directory
        self._setup_work_dir()

        # Try to calculate the feature. This will populate self.results and potentially self._err
        try:
            # self._check_requirements() ensures that the child class implements the calculate()
            # method; mypy does not recognize this, so we ignore the type error here
            self.calculate()  # type: ignore[attr-defined]
        except Exception as e:
            _errmsg = (
                f"An unexpected error occurred during the calculation of the "
                f"'{self.feature_name}' feature for the '{self.feature_type}' with index "
                f"{self.atom_bond_idx}: {e.__class__.__name__}: {e}."
            )
            if _errmsg.endswith(".."):
                _errmsg = _errmsg[:-1]
            if _errmsg.endswith(".") is False:
                _errmsg += "."
            logging.error(f"'{_namespace}' | {_loc}()\n{_errmsg}")
            raise RuntimeError(f"{_loc}(): {_errmsg}")
        else:
            # Write the results to the cache and then get the data from it
            if self._err is None:
                self._to_cache()
                self._from_cache()

        # Save the potentially generated output files and return the data
        self._save_output_files()
        return self._out, self._err

    def _check_requirements(self) -> None:
        """Check if the respective feature factory (child class) implements the required
        ``calculate()`` method and ``extraction_mode`` attribute.

        Returns
        -------
        None
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Check if child class implements calculate method
        method_names = [
            attr
            for attr in dir(self)
            if callable(getattr(self, attr)) is True and not attr.startswith("__")
        ]
        if "calculate" not in method_names:
            _errmsg = (
                f"calculate() method must be implemented in child "
                f"class '{self.__class__.__name__}'."
            )
            logging.error(f"'None' | {_loc}()\n{_errmsg}")
            raise NotImplementedError(f"{_loc}(): {_errmsg}")

        # Check if child class sets extraction_mode attribute
        if "extraction_mode" not in vars(self):
            _errmsg = (
                "Attribute 'extraction_mode' must be set in child class "
                f"'{self.__class__.__name__}', either to 'single' or 'multi'."
            )
            logging.error(f"'None' | {_loc}()\n{_errmsg}")
            raise AttributeError(f"{_loc}(): {_errmsg}")

        # Check if extraction_mode is set to either 'single' or 'multi'
        extraction_mode = str(getattr(self, "extraction_mode")).lower()
        if extraction_mode not in ["single", "multi"]:
            _errmsg = (
                f"'extraction_mode' must be set either to 'single' or 'multi', but got "
                f"'{extraction_mode}' in class '{self.__class__.__name__}'."
            )
            logging.error(f"'None' | {_loc}()\n{_errmsg}")
            raise ValueError(f"{_loc}(): {_errmsg}")

    def _from_cache(self) -> None:
        """Attempt to retrieve the requested data from the feature cache.

        If the data is found in the cache, it is stored in the ``_out`` attribute.

        ``feature_cache`` is a list of cache dictionaries for the individual conformers. The keys
        of each dictionary are the feature names, and the values are dictionaries mapping atom or bond
        indices to feature values.

        Returns
        -------
        None
        """
        if self.feature_name in self.feature_cache[self.conformer_idx]:
            if self.atom_bond_idx in self.feature_cache[self.conformer_idx][self.feature_name]:
                self._out = self.feature_cache[self.conformer_idx][self.feature_name][
                    self.atom_bond_idx
                ]

    def _to_cache(self) -> None:
        """Write the data contained in ``results`` to the feature cache.

        If the child class sets the ``extraction_mode`` attribute to "multi", this method
        expects all atom or bond indices to be present in ``results``. If indices are missing, the
        feature value is set to "_inaccessible" for all features found within ``results``. If
        certain features could not be calculated for specific atoms or bonds, those features are
        also set to "_inaccessible" for the respective indices.

        Returns
        -------
        None
        """
        # Skip if results dictionary is empty
        if self.results == {}:
            return

        # Add missing atom or bond indices to results dictionary in case no feature was calculated
        # for that specific atom or bond
        if self.extraction_mode == "multi":
            # Get all feature names present in results
            all_feature_names = set()
            for idx, data in self.results.items():
                for feature_name in data:
                    all_feature_names.add(feature_name)

            # Get all atom or bond indices of the molecule
            if self.feature_type == "atom":
                idx_list = [a.GetIdx() for a in self.mol.GetAtoms()]
            if self.feature_type == "bond":
                idx_list = [b.GetIdx() for b in self.mol.GetBonds()]

            # Add missing indices to results with value "_inaccessible" for all features found in
            # results
            for idx in idx_list:
                if idx not in self.results:
                    self.results[idx] = {}
                for feature_name in all_feature_names:
                    if feature_name not in self.results[idx]:
                        self.results[idx][feature_name] = "_inaccessible"

        # Save the results to the feature cache
        for idx, data in self.results.items():
            for feature_name, value in data.items():
                if feature_name not in self.feature_cache[self.conformer_idx]:
                    self.feature_cache[self.conformer_idx][feature_name] = {}
                self.feature_cache[self.conformer_idx][feature_name][idx] = value
