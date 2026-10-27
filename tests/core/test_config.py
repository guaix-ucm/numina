"""Functions that load and describe files, numina.core.config"""

import json

import astropy.io.fits as fits
import numpy
import pytest

import numina.core.config as cfg
import numina.core.dataload as dataload
from numina.core.dataload import DataLoaders
from numina.types.structured import BaseStructuredCalibration


class StructuredCal(BaseStructuredCalibration):
    pass


@pytest.fixture
def fits_file(tmp_path):
    filename = tmp_path / "image.fits"
    hdu = fits.PrimaryHDU(numpy.ones((2, 2)))
    hdu.header["INSTRUME"] = "TEST1"
    hdu.header["OBSMODE"] = "bias"
    hdu.writeto(filename)
    return str(filename)


@pytest.fixture
def structured_file(tmp_path):
    filename = tmp_path / "calib.json"
    StructuredCal(instrument="TEST1").writeto(filename)
    return str(filename)


def test_load_fits(fits_file):
    with cfg.load(fits_file) as hdul:
        assert numpy.all(hdul[0].data == 1)


def test_load_json(tmp_path):
    filename = tmp_path / "data.json"
    filename.write_text(json.dumps({"a": 1}))
    assert cfg.load(str(filename)) == {"a": 1}


def test_load_json_structured(structured_file):
    obj = cfg.load(structured_file)
    assert isinstance(obj, StructuredCal)
    assert obj.instrument == "TEST1"


def test_describe_fits(fits_file, tmp_path):
    assert cfg.describe(fits_file) == ("TEST1", "bias")

    filename = tmp_path / "other.fits"
    fits.PrimaryHDU().writeto(filename)
    assert cfg.describe(str(filename)) == ("unknown", "unknown")


def test_describe_json_structured(structured_file):
    assert cfg.describe(structured_file) == ("TEST1", "TBD")


def test_nothing_handles(tmp_path):
    filename = tmp_path / "data.txt"
    filename.write_text("text")
    with pytest.raises(TypeError, match="nothing handles"):
        cfg.load(str(filename))


def test_loaders_priority():
    loaders = DataLoaders()
    loaders.register("application/json", priority=20)(lambda path: "last")
    loaders.register("application/json", lambda path: path.startswith("a"), priority=5)(lambda path: "first")
    assert loaders("a.json") == "first"
    assert loaders("b.json") == "last"


@pytest.mark.xfail(reason="the functions are compared when the priority and type are equal", strict=True)
def test_loaders_same_priority():
    loaders = DataLoaders()
    loaders.register("application/json")(lambda path: "first")
    loaders.register("application/json")(lambda path: "second")
    assert loaders("a.json") == "first"


@pytest.mark.parametrize("function", [dataload.is_fits_megara, dataload.is_fits_emir])
def test_is_fits(function):
    assert function("image.fits")
    assert not function("image.json")


def test_is_json_structured(structured_file, tmp_path):
    filename = tmp_path / "data.json"
    filename.write_text(json.dumps({"a": 1}))
    assert dataload.is_json_structured(structured_file)
    assert not dataload.is_json_structured(str(filename))
