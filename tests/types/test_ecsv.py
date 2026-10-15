"""Tables in ECSV format, loaded as arrays"""

import numpy
import pytest

import numina.store
from numina.types.array import ArrayType, is_ecsv, load_ecsv_array
from numina.types.linescatalog import LinesCatalog

LINES = """\
# %ECSV 1.0
# ---
# datatype:
# - {name: wavelength, unit: Angstrom, datatype: float64}
# - {name: flag, datatype: int64}
# - {name: intensity, datatype: int64}
# - {name: comment, datatype: string}
# meta: !!omap
# - {type: LinesCatalog}
# - tags: {speclamp: ThAr, vph: LR-U}
# schema: astropy-2.0
wavelength flag intensity comment
4474.7591 1 348 "ArII from NIST"
#4478.1000  1   12 # ThI, not used
4481.8104 0 0 "ThI from NIST faint"
"""

EXTINCTION = """\
# %ECSV 1.0
# ---
# datatype:
# - {name: wavelength, unit: Angstrom, datatype: float64}
# - {name: extinction, unit: mag, datatype: float64}
# schema: astropy-2.0
wavelength extinction
3000.0 3.715
3010.0 3.1305
"""


@pytest.fixture
def lines_ecsv(tmp_path):
    path = tmp_path / "lines.ecsv"
    path.write_text(LINES)
    return str(path)


def test_is_ecsv(tmp_path, lines_ecsv):
    text = tmp_path / "lines.lis"
    text.write_text("4474.7591 1 348\n")
    assert is_ecsv(lines_ecsv)
    assert not is_ecsv(str(text))
    assert not is_ecsv(str(tmp_path / "missing.ecsv"))


def test_lines_catalog_ecsv(lines_ecsv):
    """The numeric columns, without the commented out line"""
    expected = numpy.array([[4474.7591, 1, 348], [4481.8104, 0, 0]])
    value = numina.store.load(LinesCatalog(), lines_ecsv)
    assert value.dtype == float
    numpy.testing.assert_array_equal(value, expected)


def test_lines_catalog_ecsv_as_text(tmp_path, lines_ecsv):
    """The same array as the catalog in text format"""
    text = tmp_path / "lines.lis"
    text.write_text("4474.7591 1 348 # ArII from NIST\n#4478.1000  1   12\n4481.8104 0 0 # ThI\n")
    numpy.testing.assert_array_equal(
        numina.store.load(LinesCatalog(), lines_ecsv), numina.store.load(LinesCatalog(), str(text))
    )


def test_array_type_ecsv(tmp_path):
    path = tmp_path / "extinction.ecsv"
    path.write_text(EXTINCTION)
    value = numina.store.load(ArrayType(), str(path))
    numpy.testing.assert_array_equal(value, [[3000.0, 3.715], [3010.0, 3.1305]])


def test_ecsv_without_numeric_columns(tmp_path):
    path = tmp_path / "names.ecsv"
    path.write_text(
        "# %ECSV 1.0\n# ---\n# datatype:\n# - {name: name, datatype: string}\n# schema: astropy-2.0\nname\nA\n"
    )
    with pytest.raises(ValueError, match="does not have numeric columns"):
        load_ecsv_array(str(path))
