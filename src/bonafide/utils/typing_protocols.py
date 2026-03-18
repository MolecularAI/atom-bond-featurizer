"""Typing protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Union

if TYPE_CHECKING:
    from rdkit import Chem


class _MultiwfnMixinProtocol(Protocol):
    """Structural subtyping for Multiwfn mixin classes."""

    feature_name: str
    multiplicity: int
    mol: Chem.rdchem.Mol
    results: Dict[int, Dict[str, Optional[Union[int, float, bool, str]]]]

    # Implemented in multiwfn_population_analysis.py (_Multiwfn3DAtomPopulationAnalysis)
    def _run_multiwfn(self, command_list: List[Union[int, float, str]]) -> None: ...
    def _read_output_file3(self, scheme_name: str) -> None: ...
