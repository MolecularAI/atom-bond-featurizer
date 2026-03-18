"""Initialization."""

import os
import resource
import warnings
from importlib.metadata import version
from pathlib import Path

from rdkit import RDLogger

# Disable warnings
warnings.filterwarnings(action="ignore")
RDLogger.DisableLog("rdApp.*")

# Set version
try:
    __version__ = version(distribution_name="bonafide")
except Exception:
    __version__ = "unknown"

# Increase stack size limit (required for large molecules)
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

# Add environment's bin directory to PATH (only relevant in case BONAFIDE is called from outside
# its environment)
_bin_path = Path(__file__).parents[4] / "bin"
if _bin_path.exists():
    _bin_path_str = str(_bin_path)
    _current_path = os.environ.get("PATH", "")
    if _bin_path_str not in _current_path:
        os.environ["PATH"] = f"{_current_path}:{_bin_path_str}"

# Import main classes
__all__ = ["AtomBondFeaturizer", "LogFileAnalyzer"]
from bonafide.bonafide import AtomBondFeaturizer
from bonafide.log_file_analysis import LogFileAnalyzer
