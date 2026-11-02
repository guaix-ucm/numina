"""Envelope of the self-describing calibration files"""

import json

import numpy
import pytest
from astropy.io import fits

import numina.store
from numina.schemas import SchemaValidationError
from numina.store.envelope import FormatError, check_envelope, container, read_ecsv_meta, read_envelope
from numina.types.array import ArrayType
from numina.types.frame import DataFrameType
from numina.types.linescatalog import LinesCatalog
from numina.types.product import DataProductMixin

LINES = """\
# %ECSV 1.0
# ---
# datatype:
# - {{name: wavelength, unit: Angstrom, datatype: float64}}
# - {{name: comment, datatype: string}}
# - {{name: intensity, datatype: int64}}
# - {{name: flag, datatype: int64}}
# meta: !!omap
# - {{envelope: 1}}
# - {{type: {type}}}
# - {{format: {format}}}
# - {{instrument: MEGARA}}
# - {{uuid: 21cdd29f-f143-4686-ae49-8df9b8ed7be8}}
# - tags: {{speclamp: ThAr, vph: LR-U}}
# - validity: {{start: 2017-01-01}}
# - {{description: LR-U}}
# schema: astropy-2.0
wavelength comment intensity flag
4474.7591 "ArII from NIST" 348 1
4481.8104 "ThI from NIST faint" 0 0
"""

NO_ENVELOPE = """\
# %ECSV 1.0
# ---
# datatype:
# - {name: wavelength, unit: Angstrom, datatype: float64}
# - {name: extinction, unit: mag, datatype: float64}
# meta: !!omap
# - {type: ReferenceExtinctionTable}
# schema: astropy-2.0
wavelength extinction
3000.0 3.715
3010.0 3.1305
"""


def write_lines(tmp_path, type="LinesCatalog", format=2):
    path = tmp_path / "lines.ecsv"
    path.write_text(LINES.format(type=type, format=format))
    return str(path)


def write_fits(tmp_path, name="bpm.fits.gz", **keywords):
    path = tmp_path / name
    hdu = fits.PrimaryHDU(numpy.zeros((2, 3), dtype="int16"))
    hdu.header["INSTRUME"] = "MEGARA"
    hdu.header["UUID"] = "b019c7cf-de6b-48c4-9cc3-d0d800a3aa72"
    hdu.header["INSCONF"] = "9f3fecbf-376c-47ae-a188-313b7d829104"
    for key, value in keywords.items():
        hdu.header[key] = value
    hdu.writeto(path)
    return str(path)


class MasterBPM(DataProductMixin, DataFrameType):
    pass


class ExtinctionTable(DataProductMixin, ArrayType):
    ecsv_columns = ("wavelength", "extinction")


def test_container(tmp_path):
    ecsv = write_lines(tmp_path)
    fits_gz = write_fits(tmp_path)
    fits_plain = write_fits(tmp_path, name="bpm.fits")
    jfile = tmp_path / "trace.json"
    jfile.write_text('\n  {"meta": {}}')
    text = tmp_path / "lines.lis"
    text.write_text("4474.7591 1 348\n")
    assert container(ecsv) == "ecsv"
    assert container(fits_gz) == "fits"
    assert container(fits_plain) == "fits"
    assert container(jfile) == "json"
    assert container(text) is None


def test_read_ecsv_envelope(tmp_path):
    """The dates are strings, as in the schema"""
    envelope = read_envelope(write_lines(tmp_path))
    assert envelope == {
        "envelope": 1,
        "type": "LinesCatalog",
        "format": 2,
        "instrument": "MEGARA",
        "uuid": "21cdd29f-f143-4686-ae49-8df9b8ed7be8",
        "tags": {"speclamp": "ThAr", "vph": "LR-U"},
        "validity": {"start": "2017-01-01"},
        "description": "LR-U",
    }


def test_ecsv_without_envelope(tmp_path):
    path = tmp_path / "extinction.ecsv"
    path.write_text(NO_ENVELOPE)
    assert read_ecsv_meta(path) == {"type": "ReferenceExtinctionTable"}
    assert read_envelope(path) is None


def test_read_fits_envelope(tmp_path):
    path = write_fits(tmp_path, NUMENV=1, NUMTYPE="MasterBPM", NUMFMT=1)
    assert read_envelope(path) == {
        "envelope": 1,
        "type": "MasterBPM",
        "format": 1,
        "instrument": "MEGARA",
        "uuid": "b019c7cf-de6b-48c4-9cc3-d0d800a3aa72",
        "profile": "9f3fecbf-376c-47ae-a188-313b7d829104",
        "tags": {},
    }


def test_fits_without_numenv(tmp_path):
    """The products of the recipes have NUMTYPE, but no envelope"""
    path = write_fits(tmp_path, NUMTYPE="MasterBPM")
    assert read_envelope(path) is None


def test_read_json_envelope(tmp_path):
    meta = {"envelope": 1, "type": "TraceMap", "format": 2, "instrument": "MEGARA", "uuid": "1", "tags": {}}
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"meta": meta, "data": {"contents": []}}))
    assert read_envelope(path) == meta
    # A structured calibration of format 1, flat
    path.write_text(json.dumps({"type": "TraceMap", "tags": {}, "meta_info": {}}))
    assert read_envelope(path) is None


def test_invalid_envelope(tmp_path):
    path = write_fits(tmp_path, NUMENV=1, NUMFMT=1)
    with pytest.raises(SchemaValidationError, match=r"bpm.fits.gz: invalid envelope file, at top level: 'type'"):
        read_envelope(path)


def test_load_format_2(tmp_path):
    """The columns are read by name, in the order of ecsv_columns"""
    path = write_lines(tmp_path)
    value = numina.store.load(LinesCatalog(), path)
    numpy.testing.assert_array_equal(value, [[4474.7591, 1, 348], [4481.8104, 0, 0]])


def test_load_format_1_text(tmp_path):
    path = tmp_path / "lines.lis"
    path.write_text("4474.7591 1 348\n4481.8104 0 0\n")
    value = numina.store.load(LinesCatalog(), str(path))
    numpy.testing.assert_array_equal(value, [[4474.7591, 1, 348], [4481.8104, 0, 0]])


def test_load_unknown_format(tmp_path):
    path = write_lines(tmp_path, format=3)
    msg = r"lines.ecsv: LinesCatalog in format 3, this version reads formats 1, 2; update numina"
    with pytest.raises(FormatError, match=msg):
        numina.store.load(LinesCatalog(), path)


def test_load_other_type(tmp_path):
    path = write_lines(tmp_path, type="ReferenceExtinctionTable")
    with pytest.raises(FormatError, match=r"lines.ecsv: the file is a ReferenceExtinctionTable, not a LinesCatalog"):
        numina.store.load(LinesCatalog(), path)


def test_array_reads_any_type(tmp_path):
    """An ArrayType is not a product, it reads the numeric columns of any table"""
    path = write_lines(tmp_path)
    value = numina.store.load(ArrayType(), path)
    numpy.testing.assert_array_equal(value, [[4474.7591, 348, 1], [4481.8104, 0, 0]])


def test_array_columns(tmp_path):
    path = tmp_path / "extinction.ecsv"
    path.write_text(NO_ENVELOPE.replace("# - {type: ReferenceExtinctionTable}", ENVELOPE_EXT))
    value = numina.store.load(ExtinctionTable(), str(path))
    numpy.testing.assert_array_equal(value, [[3000.0, 3.715], [3010.0, 3.1305]])


ENVELOPE_EXT = """\
# - {envelope: 1}
# - {type: ExtinctionTable}
# - {format: 2}
# - {instrument: MEGARA}
# - {uuid: 53f22da6-cebb-4112-8489-c14e6e7a26a7}
# - tags: {}"""


def test_array_missing_columns(tmp_path):
    path = tmp_path / "extinction.ecsv"
    text = NO_ENVELOPE.replace("# - {type: ReferenceExtinctionTable}", ENVELOPE_EXT)
    path.write_text(
        text.replace("name: extinction", "name: airmass").replace("wavelength extinction", "wavelength airmass")
    )
    with pytest.raises(ValueError, match="the table does not have the columns extinction"):
        numina.store.load(ExtinctionTable(), str(path))


def test_load_fits(tmp_path):
    path = write_fits(tmp_path, NUMENV=1, NUMTYPE="MasterBPM", NUMFMT=1)
    value = numina.store.load(MasterBPM(), path)
    assert value.filename == path
    path2 = write_fits(tmp_path, name="other.fits", NUMENV=1, NUMTYPE="MasterBPM", NUMFMT=2)
    with pytest.raises(FormatError, match="MasterBPM in format 2, this version reads formats 1; update numina"):
        numina.store.load(MasterBPM(), path2)


def test_owner_of_the_formats(tmp_path):
    """The error names the package of the class that declares the formats"""

    class Catalog(LinesCatalog):
        __numina_formats__ = {1: "_datatype_load"}

        def name(self):
            return "LinesCatalog"

    Catalog.__module__ = "megaradrp.ntypes"
    path = write_lines(tmp_path)
    with pytest.raises(FormatError, match="reads formats 1; update megaradrp"):
        check_envelope(Catalog(), path, read_envelope(path))
