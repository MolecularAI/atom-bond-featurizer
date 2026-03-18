"""Oxidation state feature."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict

from rdkit import Chem

from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.helper_functions import get_function_or_method_name
from bonafide.utils.helper_functions_chemistry import from_periodic_table

if TYPE_CHECKING:
    from mendeleev import element


class Bonafide2DAtomOxidationState(BaseFeaturizer):
    """Feature factory for the 2D atom feature "oxidation_state", implemented within this
    package.

    The index of this feature is 36 (see the ``list_atom_features()`` and
    ``list_bond_features()`` method). The corresponding configuration settings can be found
    under "bonafide.oxidation_state" in the _feature_config.toml file.
    """

    _periodic_table: Dict[str, element]
    en_scale: str

    def __init__(self) -> None:
        self.extraction_mode = "single"
        super().__init__()

    def calculate(self) -> None:
        """Calculate the ``bonafide2D-atom-oxidation_state`` feature."""
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Get electronegativity for the atom under consideration
        atom = self.mol.GetAtomWithIdx(self.atom_bond_idx)
        _, element_data = from_periodic_table(
            periodic_table=self._periodic_table, element_symbol=atom.GetSymbol()
        )
        atom_en = element_data.electronegativity(scale=self.en_scale)

        # Get the data from the neighbors
        contributions = []
        for neighbor in atom.GetNeighbors():
            _, element_data_neighbor = from_periodic_table(
                periodic_table=self._periodic_table, element_symbol=neighbor.GetSymbol()
            )
            neighbor_en = element_data_neighbor.electronegativity(scale=self.en_scale)
            neighbor_bond = self.mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            neighbor_bond_order = neighbor_bond.GetBondTypeAsDouble()
            if neighbor_en > atom_en:
                contributions.append(neighbor_bond_order)
            elif neighbor_en < atom_en:
                contributions.append(-neighbor_bond_order)

        # Calculate the oxidation state and save the results
        value = int(sum(contributions) + atom.GetFormalCharge())
        self.results[self.atom_bond_idx] = {self.feature_name: value}

        # Log a warning if oxidation states are calculated without hydrogen atoms
        _helper_mol = Chem.Mol(self.mol)
        _helper_mol = Chem.AddHs(_helper_mol)
        if _helper_mol.GetNumAtoms() != self.mol.GetNumAtoms():
            _namespace = self.conformer_name[::-1].split("__", 1)[-1][::-1]
            logging.warning(
                f"'{_namespace}' | {_loc}()\nThe '{self.feature_name}' feature was calculated "
                f"for atom with index '{self.atom_bond_idx}' without adding hydrogen atoms to the "
                "molecule. Check if this is desired."
            )
