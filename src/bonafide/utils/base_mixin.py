"""Mixin class with common base functionality for ``BaseFeaturizer`` and ``BaseSinglePoint``."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from bonafide.utils.helper_functions import get_function_or_method_name

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class _BaseMixin:
    """Set up a temporary working directory before the feature or single-point energy calculation
    and save the output files after the calculation is done.

    Attributes
    ----------
    _keep_output_files : bool
        If ``True``, all output files created during the feature calculations are kept. If
        ``False``, they are removed when the calculation is done.
    conformer_name : str
        The name of the conformer for which the feature is requested.
    work_dir_name : Optional[str]
        The name of the working directory where temporary files are stored during feature
        calculation.
    """

    _keep_output_files: bool
    conformer_name: str
    work_dir_name: str

    # Common attributes for child classes
    charge: Optional[int]
    coordinates: Optional[NDArray[np.float64]]
    electronic_struc_n: Optional[str]
    electronic_struc_n_plus1: Optional[str]
    electronic_struc_n_minus1: Optional[str]
    elements: NDArray[np.str_]
    global_feature_cache: List[Dict[str, Optional[Union[str, bool, int, float]]]]
    multiplicity: Optional[int]

    def _setup_work_dir(self) -> None:
        """Set up the temporary working directory for a feature or single-point energy calculation.

        The temporary working directory is set up inside the output files directory. If the user
        did not request an output files directory, ``_output_directory`` is set to the current
        working directory (in which the working directory is then created).

        Returns
        -------
        None
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Define working directory name
        self.work_dir_name = (
            f"_w__{self.conformer_name}__"
            f"{datetime.now().strftime('%Y%m%d%H%M%S%f')[2:]}-{uuid.uuid4().hex[:6]}"
        )

        # Check if the directory already exists
        if os.path.exists(self.work_dir_name):
            _errmsg = (
                f"Temporary working directory path at "
                f"'{os.path.abspath(self.work_dir_name)}' already exists."
            )
            _namespace = self.conformer_name[::-1].split("__", 1)[-1][::-1]
            logging.error(f"'{_namespace}' | {_loc}()\n{_errmsg}")
            raise FileExistsError(f"{_loc}(): {_errmsg}")

        # Create directory and change to it
        os.mkdir(self.work_dir_name)
        os.chdir(self.work_dir_name)

    def _save_output_files(self) -> None:
        """Save the potentially generated output files during a feature or single-point energy
        calculation and delete the temporary working directory.

        The child classes (feature factories) are responsible for deciding which files to
        preserve. If ``_keep_output_files`` is ``False``, no output files are saved.

        Returns
        -------
        None
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Leave working directory
        os.chdir("..")

        # Save output files if requested by the user
        if self._keep_output_files is True:
            for item in os.listdir(self.work_dir_name):
                try:
                    source = os.path.join(self.work_dir_name, item)
                    dest = os.path.join(os.getcwd(), item)
                    if os.path.isfile(source):
                        shutil.copy2(source, dest)
                    if os.path.isdir(source):
                        shutil.copytree(source, dest)
                except Exception as e:
                    _errmsg = f"Could not copy '{source}' to '{dest}': {e}."
                    _namespace = self.conformer_name[::-1].split("__", 1)[-1][::-1]
                    logging.error(f"'{_namespace}' | {_loc}()\n{_errmsg}")
                    raise IOError(f"{_loc}(): {_errmsg}")

        # Delete working directory
        shutil.rmtree(self.work_dir_name)
