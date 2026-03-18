"""Test functions for the ``bonafide.utils.environment`` module."""

import copy
import os
from typing import Generator

import pytest

from bonafide.utils.environment import Environment

_TEST_VARIABLE = "TEST_VAR"
_CONST_VARIABLE = "CONST_VARIABLE"


@pytest.fixture(autouse=True)
def maintain_fresh_environment() -> Generator[None, None, None]:
    """Reset environment after each test function."""
    original_env = copy.deepcopy(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


###########################################
# Tests for the set_environment() method. #
###########################################


@pytest.mark.set_environment
def test_set_environment() -> None:
    """Test for the ``set_environment()`` method: set new variable."""
    assert _TEST_VARIABLE not in os.environ

    _val = "new_value"
    env = Environment(TEST_VAR=_val)
    env.set_environment()
    assert os.environ["TEST_VAR"] == _val


@pytest.mark.set_environment
def test_set_environment2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test for the ``set_environment()`` method: overwrite existing variable."""
    assert _TEST_VARIABLE not in os.environ

    # Prepare environment
    _val_old = "old_value"
    monkeypatch.setenv(_TEST_VARIABLE, _val_old)
    assert os.environ[_TEST_VARIABLE] == _val_old

    # Update environment
    _val_new = "new_value"
    env = Environment(TEST_VAR=_val_new)
    env.set_environment()
    assert os.environ[_TEST_VARIABLE] == _val_new


@pytest.mark.set_environment
def test_set_environment3() -> None:
    """Test for the ``set_environment()`` method: don't set variable if value is None."""
    assert _TEST_VARIABLE not in os.environ

    env = Environment(TEST_VAR=None)
    env.set_environment()
    assert _TEST_VARIABLE not in os.environ


#############################################
# Tests for the reset_environment() method. #
#############################################


@pytest.mark.reset_environment
def test_reset_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test for the ``reset_environment()`` method: one value."""
    assert _TEST_VARIABLE not in os.environ
    assert _CONST_VARIABLE not in os.environ

    _val_old = "old_value"
    _val_constant = "constant_value"

    monkeypatch.setenv(_TEST_VARIABLE, _val_old)
    monkeypatch.setenv(_CONST_VARIABLE, _val_constant)

    assert os.environ[_TEST_VARIABLE] == _val_old
    assert os.environ[_CONST_VARIABLE] == _val_constant

    _val_new = "new_value"
    env = Environment(TEST_VAR=_val_new)
    env.set_environment()
    assert os.environ[_TEST_VARIABLE] == _val_new
    assert os.environ[_CONST_VARIABLE] == _val_constant

    env.reset_environment()
    assert os.environ[_TEST_VARIABLE] == _val_old
    assert os.environ[_CONST_VARIABLE] == _val_constant


@pytest.mark.reset_environment
def test_reset_environment2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test for the ``reset_environment()`` method: multiple values."""
    assert _TEST_VARIABLE not in os.environ
    assert _CONST_VARIABLE not in os.environ
    assert "TEST_VAR2" not in os.environ
    assert "TEST_VAR3" not in os.environ
    assert "TEST_VAR4" not in os.environ

    _val_old = "old_value"
    _val_constant = "constant_value"

    monkeypatch.setenv(_TEST_VARIABLE, _val_old)
    monkeypatch.setenv(_CONST_VARIABLE, _val_constant)

    assert os.environ[_TEST_VARIABLE] == _val_old
    assert os.environ[_CONST_VARIABLE] == _val_constant

    _val_new = "new_value"
    env = Environment(TEST_VAR=_val_new, TEST_VAR2=5, TEST_VAR3="another_new_value", TEST_VAR4=None)
    env.set_environment()
    assert os.environ[_TEST_VARIABLE] == _val_new
    assert os.environ["TEST_VAR2"] == "5"
    assert os.environ["TEST_VAR3"] == "another_new_value"
    assert "TEST_VAR4" not in os.environ
    assert os.environ[_CONST_VARIABLE] == _val_constant

    env.reset_environment()
    assert os.environ[_TEST_VARIABLE] == _val_old
    assert os.environ[_CONST_VARIABLE] == _val_constant
    assert "TEST_VAR2" not in os.environ
    assert "TEST_VAR3" not in os.environ
    assert "TEST_VAR4" not in os.environ
