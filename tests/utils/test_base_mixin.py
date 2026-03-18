"""Test functions for the ``bonafide.utils.base_mixin`` module."""

import logging
import os
import shutil
from pathlib import Path

import pytest

from bonafide.utils.base_mixin import _BaseMixin


class DummyClass(_BaseMixin):
    """Dummy class to test the _BaseMixin methods."""

    def __init__(self, _keep_output_files: bool) -> None:
        self.conformer_name = "irrelevant"
        self._keep_output_files = _keep_output_files


###########################################
# Tests for the _setup_work_dir() method. #
###########################################


@pytest.mark.setup_work_dir
def test__setup_work_dir(caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``_setup_work_dir()`` method."""
    _init_cwd = os.getcwd()
    _init_dir_contents = os.listdir()
    _init_dir_contents.sort()

    # Create working directory
    d = DummyClass(_keep_output_files=True)
    d._setup_work_dir()
    workdir_path = Path(d.work_dir_name)
    assert os.getcwd() == str(Path(_init_cwd / workdir_path))

    os.chdir("..")
    assert workdir_path.exists()
    assert workdir_path.is_dir()

    _new_dir_contents = os.listdir()
    _new_dir_contents.sort()

    _expected_dir_contents = [x for x in _init_dir_contents]
    _expected_dir_contents.append(d.work_dir_name)
    _expected_dir_contents.sort()

    assert _new_dir_contents == _expected_dir_contents

    # Check logging
    assert len(caplog.records) == 0

    # Clean up
    shutil.rmtree(d.work_dir_name)


##############################################
# Tests for the _save_output_files() method. #
##############################################


@pytest.mark.save_output_files
@pytest.mark.parametrize("keep_files", [True, False])
def test__save_output_files(caplog: pytest.LogCaptureFixture, keep_files: bool) -> None:
    """Test for the ``_save_output_files()`` method: valid examples."""
    _init_cwd = os.getcwd()
    _init_dir_contents = os.listdir()
    _init_dir_contents.sort()

    # Setup working directory
    d = DummyClass(_keep_output_files=keep_files)
    d._setup_work_dir()
    workdir_path = Path(d.work_dir_name)

    # Create dummy output data
    os.mkdir("test_output_dir")
    with open("test_file.txt", "w") as f:
        f.write("This is a test output file.")
    with open(Path("test_output_dir") / "test_file_in_dir.out", "w") as f:
        f.write("This is a test output file inside a directory.")

    # Save output files
    d._save_output_files()
    assert os.getcwd() == _init_cwd
    assert workdir_path.exists() is False

    _new_dir_contents = os.listdir()
    _new_dir_contents.sort()

    # Check that no files were saved (if not requested)
    if keep_files is False:
        assert _new_dir_contents == _init_dir_contents
        return

    # Check that data was saved (if requested)
    assert len(_new_dir_contents) == len(_init_dir_contents) + 2
    assert "test_file.txt" in _new_dir_contents
    assert "test_output_dir" in _new_dir_contents
    assert Path(os.getcwd() / Path("test_file.txt")).exists()
    assert Path(os.getcwd() / Path("test_output_dir")).exists()
    assert Path(os.getcwd() / Path("test_output_dir") / "test_file_in_dir.out").exists()

    # Check logging
    assert len(caplog.records) == 0

    # Clean up
    os.remove("test_file.txt")
    shutil.rmtree("test_output_dir")


@pytest.mark.save_output_files
def test__save_output_files2(caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``_save_output_files()`` method: invalid examples."""
    os.mkdir("i_am_already_here")
    _init_dir_contents = os.listdir()
    _init_dir_contents.sort()

    # Setup working directory
    d = DummyClass(_keep_output_files=True)
    d._setup_work_dir()

    # Create dummy output data
    os.mkdir("i_am_already_here")
    with open(Path("i_am_already_here") / "test_file_in_dir.out", "w") as f:
        f.write("This is a test output file inside a directory.")

    # Save output files
    with pytest.raises(IOError, match="Could not copy '"):
        d._save_output_files()

    _new_dir_contents = os.listdir()
    _new_dir_contents.sort()

    _expected_dir_contents = [x for x in _init_dir_contents]
    _expected_dir_contents.append(d.work_dir_name)
    _expected_dir_contents.sort()

    assert _new_dir_contents == _expected_dir_contents
    assert os.listdir("i_am_already_here") == []

    # Check logging
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR

    # Clean up
    shutil.rmtree("i_am_already_here")
    shutil.rmtree(d.work_dir_name)
