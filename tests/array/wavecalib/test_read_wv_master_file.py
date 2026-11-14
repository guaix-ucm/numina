"""Tests for numina.array.wavecalib.core.read_wv_master_file"""

from astropy.table import Table
import numpy as np
import pytest

from numina.array.wavecalib.core import read_wv_master_file

WAVELENGTHS = [4000.0, 5000.0, 6000.0, 7000.0]
FLAGS = [1, 0, 1, 0]
INTENSITIES = [10.0, 1.0, 5.0, 0.5]
BRIGHTEST = [4000.0, 6000.0]


@pytest.fixture
def ecsv_file(tmp_path):
    filename = tmp_path / "lines.ecsv"
    table = Table([WAVELENGTHS, FLAGS, INTENSITIES], names=["wavelength", "flag", "intensity"])
    table.write(filename, format="ascii.ecsv")
    return filename


@pytest.mark.parametrize("lines, expected", [("brightest", BRIGHTEST), ("all", WAVELENGTHS)])
def test_ecsv(ecsv_file, lines, expected):
    wv_master = read_wv_master_file(str(ecsv_file), lines=lines)
    assert isinstance(wv_master, np.ndarray)
    assert np.allclose(wv_master, expected)


def test_ecsv_pathlib_and_uppercase_extension(ecsv_file):
    filename = ecsv_file.rename(ecsv_file.with_suffix(".ECSV"))
    wv_master = read_wv_master_file(filename, lines="brightest")
    assert np.allclose(wv_master, BRIGHTEST)


@pytest.mark.parametrize("lines, expected", [("brightest", BRIGHTEST), ("all", WAVELENGTHS)])
def test_txt_three_columns(tmp_path, lines, expected):
    filename = tmp_path / "lines.txt"
    np.savetxt(filename, np.column_stack([WAVELENGTHS, FLAGS, INTENSITIES]))
    wv_master = read_wv_master_file(str(filename), lines=lines)
    assert np.allclose(wv_master, expected)


@pytest.mark.parametrize("lines", ["brightest", "all"])
def test_txt_two_columns(tmp_path, lines):
    # old format (wavelength, intensity): all lines are returned
    filename = tmp_path / "lines.txt"
    np.savetxt(filename, np.column_stack([WAVELENGTHS, INTENSITIES]))
    wv_master = read_wv_master_file(str(filename), lines=lines)
    assert np.allclose(wv_master, WAVELENGTHS)


@pytest.mark.parametrize("lines", ["brightest", "all"])
def test_txt_one_column(tmp_path, lines):
    filename = tmp_path / "lines.txt"
    np.savetxt(filename, WAVELENGTHS)
    wv_master = read_wv_master_file(str(filename), lines=lines)
    assert np.allclose(wv_master, WAVELENGTHS)


def test_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_wv_master_file(str(tmp_path / "missing.ecsv"))


def test_invalid_lines(ecsv_file):
    with pytest.raises(ValueError, match="Unexpected lines"):
        read_wv_master_file(str(ecsv_file), lines="faint")
