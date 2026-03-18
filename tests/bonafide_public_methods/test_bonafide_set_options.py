"""Test functions for the ``set_options()`` method in the ``bonafide.bonafide`` module."""

from __future__ import annotations

import copy
import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

import pytest
from rdkit import RDLogger

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

# Disable RDKit logging
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def _get_config_setting(config_dict: Dict[str, Any], config_path: str) -> Any:
    """Helper function to get a setting from a nested dictionary using a dot-separated path."""
    keys = config_path.split(".")
    value = config_dict
    for key in keys:
        try:
            value = value[key]
        except KeyError:
            try:
                value = value[key.lower()]
            except Exception as e:
                raise RuntimeError(
                    f"FROM TEST SCRIPT: Key '{key}' not found in configuration dictionary: {e}."
                )
    return value


# Everything before the last . in configs is case-insensitive, the last part is case-sensitive
@pytest.mark.set_options
@pytest.mark.parametrize(
    "configs, error_expected, error_type",
    [
        # Valid examples (must all change default settings)
        ([("Bonafide.autocorrelation.depth", 3)], False, None),
        (
            [("dbstep.r", 2.8), ("dbstep.noH", True), ("morfeus.sasa.radii_type", "bondi")],
            False,
            None,
        ),
        (
            [("xtb.OMP_STACKSIZE", "2g"), ("xtb.acc", 0.5)],
            False,
            None,
        ),
        (
            [
                ("multiwfn.NUM_THREADS", 8),
                ("multiwfn.BOND_analysis.ibsi_grid", "ultrafine"),
            ],
            False,
            None,
        ),
        (
            [
                ("multiwfn.bond_analysis.ibsi_grid", "medium"),
                ("multiwfn.fuzzy.n_iterations_becke_partition", 5),
            ],
            False,
            None,
        ),
        # Invalid examples
        (
            "wrong_input_type",
            True,
            TypeError,
        ),
        (
            [],
            True,
            ValueError,
        ),
        (
            [123],
            True,
            TypeError,
        ),
        (
            [("bonafide.autocorrelation.depth",), 3],
            True,
            ValueError,
        ),
        (
            [("bonafide.Autocorrelation.depth", 3, "i_should_not_be_here")],
            True,
            ValueError,
        ),
        (
            [(["bonafide.autocorrelation.depth"], 3)],
            True,
            TypeError,
        ),
        ([("Bonafide.i_dont_exist.depth", 3)], True, ValueError),
        ([("bonafide.autocorrelation.the_real_depth", 3)], True, ValueError),
        ([("I_dont_exist", True)], True, ValueError),
        ([("Bonafide.autocorrelation.depth", False)], True, "always ValueError"),
        ([("XTB.solvent", "not_a_solvent_for_sure")], True, "always ValueError"),
        ([("rdkit.fingerprint.fpSize", -10)], True, "always ValueError"),
    ],
)
def test_set_options(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: AtomBondFeaturizer,
    clean_up_logfile: Callable[[str], None],
    configs: List[Tuple[str, Any]],
    error_expected: bool,
    error_type: Any,
) -> None:
    """Test for the ``set_options()`` method."""
    f = fresh_featurizer()

    # Valid cases
    if error_expected is False:
        # Get the initial (default settings)
        _init_settings = []
        for request in configs:
            _val = _get_config_setting(config_dict=f._feature_config, config_path=request[0])
            _init_settings.append(_val)
            assert _val != request[1]

        # Change configuration settings
        f.set_options(configs=configs)

        # Check that the settings have changed
        for idx, request in enumerate(configs):
            _new_setting = _get_config_setting(
                config_dict=f._feature_config, config_path=request[0]
            )
            assert _new_setting == request[1]
            assert _new_setting != _init_settings[idx]

        # Check logs
        assert len(caplog.records) > 0
        assert all(record.levelno == logging.INFO for record in caplog.records)

        # Clean up
        clean_up_logfile()
        return

    # Error cases
    _default_config_settings = copy.deepcopy(f._feature_config)

    if type(error_type) != str:
        with pytest.raises(error_type, match="Invalid input to 'configs':"):
            f.set_options(configs=configs)
    else:
        with pytest.raises(ValueError, match="Incorrect data encountered in"):
            f.set_options(configs=configs)

    assert f._feature_config == _default_config_settings

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)

    # Clean up
    clean_up_logfile()
