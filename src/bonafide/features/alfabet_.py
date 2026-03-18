"""Bond dissociation energy features from ``ALFABET``."""

import logging
from typing import Dict

import pandas as pd

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.driver import external_driver
from bonafide.utils.helper_functions import get_function_or_method_name
from bonafide.utils.helper_functions_chemistry import get_atom_bond_mapping_dicts


class _Alfabet2DBond(BaseFeaturizer):
    """Parent feature factory for the 2D atom ALFABET features.

    For details, please refer to the ALFABET repository (https://github.com/NREL/alfabet,
    last accessed on 09.09.2025).
    """

    python_interpreter_path: str

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``alfabet2D-bond-bond_dissociation_energy`` and
        ``alfabet2D-bond-bond_dissociation_free_energy`` feature."""
        # Get the canonical SMILES string and the bond mapping dictionary to ensure that ALFABET
        # is run with the canonical SMILES string to avoid potential issues with different
        # atom/bond orderings.
        _, mapping_dict_bonds, canonical_smiles = get_atom_bond_mapping_dicts(self.mol)

        # ALFABET is run in its separate Python environment through a helper script that is
        # temporarily created and run with the respective Python interpreter. This was necessary
        # because ALFABET was not compatible with BONAFIDE's python environment.

        # Python script for ALFABET
        alfabet_script = [
            "import pandas as pd",
            "import os",
            "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'",
            "from alfabet import model",
            f"df = model.predict([r'{canonical_smiles}'])",
            f"df.to_csv('Alfabet2DBond_{self.conformer_name}.csv', index=False)",
        ]
        alfabet_script_str = "\n".join(alfabet_script)

        # Run ALFABET
        res = external_driver(
            program_path=self.python_interpreter_path,
            program_input=alfabet_script_str,
            input_file_extension=".py",
            namespace=self.conformer_name[::-1].split("__", 1)[-1][::-1],
            dependencies=["pandas", "alfabet"],
            capture_output=True,
            text=True,
            check=False,
        )

        # Check for errors
        stderr = res.stderr
        returncode = res.returncode
        if returncode != 0:
            self._err = f"returncode: {returncode}, stderr: {stderr}"
            return

        # Save the results
        self._read_output_file(mapping_dict=mapping_dict_bonds)

    def _read_output_file(self, mapping_dict: Dict[int, int]) -> None:
        """Read the ALFABET output pandas DataFrame and write the results to the results
        dictionary.

        Only the bonds that can be predicted by ALFABET will have an entry in the DataFrame. If
        molecules with no hydrogen atoms added are passed to BONAFIDE, the X-H dissociation
        energies still will be predicted by ALFABET, but the results will not appear in the final
        BONAFIDE output, as the bonds do not exist in the actual input molecule. Add hydrogen atoms
        to the molecule before passing it to BONAFIDE to avoid this.

        Parameters
        ----------
        mapping_dict : Dict[int, int]
            The mapping dictionary to map the bond indices from the canonical SMILES string to the
            bond indices of the input molecule. This is included for security to ensure that the
            bond indices are handled correctly.

        Returns
        -------
        None
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Read the output file
        df = pd.read_csv(f"Alfabet2DBond_{self.conformer_name}.csv")

        # Get the data and write it to the results dictionary
        for _, row_data in df.iterrows():
            bond_idx = int(row_data["bond_index"])
            bde = row_data["bde_pred"]
            bdfe = row_data["bdfe_pred"]
            valid = row_data["is_valid"]

            if valid is False:
                _namespace = self.conformer_name[::-1].split("__", 1)[-1][::-1]
                logging.warning(
                    f"'{_namespace}' | {_loc}()\nPrediction of the bond dissociation (free) "
                    f"energy with ALFABET for bond with index {bond_idx} was labeled as invalid. "
                    "Check your input and the output."
                )

            if bond_idx in mapping_dict:
                self.results[mapping_dict[bond_idx]] = {
                    "alfabet2D-bond-bond_dissociation_energy": bde,
                    "alfabet2D-bond-bond_dissociation_free_energy": bdfe,
                }


class Alfabet2DBondBondDissociationEnergy(_Alfabet2DBond):
    """Feature factory for the 2D bond feature "bond_dissociation_energy", calculated with
    alfabet.

    The index of this feature is 0 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "alfabet" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Alfabet2DBond


class Alfabet2DBondBondDissociationFreeEnergy(_Alfabet2DBond):
    """Feature factory for the 2D bond feature "bond_dissociation_free_energy", calculated with
    alfabet.

    The index of this feature is 1 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "alfabet" in the _feature_config.toml file.
    """

    def __init__(self) -> None:
        super().__init__()

    # This feature is automatically calculated in _Alfabet2DBond
