"""Correctors of numina.processing"""

import logging

import astropy.io.fits as fits
import numpy
import pytest

import numina.processing as proc
from numina.datamodel import DataModel

UUID = "1c9b6a4e-7d2f-4b3a-9e8c-0f5d2a7b6c31"


def image(value, shape=(4, 5), dtype="float32", variance=None, **keywords):
    hdu = fits.PrimaryHDU(numpy.full(shape, value, dtype=dtype))
    hdu.header["UUID"] = UUID
    for key, val in keywords.items():
        hdu.header[key] = val
    hdul = fits.HDUList([hdu])
    if variance is not None:
        hdul.append(fits.ImageHDU(numpy.full(shape, variance, dtype=dtype), name="VARIANCE"))
    return hdul


def test_promote_hdu():
    primary = fits.PrimaryHDU(numpy.ones((2, 2), dtype="uint16"))
    primary.header["OBSERVER"] = "me"
    extension = fits.ImageHDU(numpy.ones((2, 2), dtype="int16"), name="EXT")
    table = fits.BinTableHDU.from_columns([fits.Column(name="a", format="J", array=[1, 2])])
    empty = fits.ImageHDU()

    result = proc.promote_hdulist(fits.HDUList([primary, extension, table, empty]))

    assert isinstance(result[0], fits.PrimaryHDU)
    assert result[0].data.dtype == numpy.float32
    assert result[0].header["OBSERVER"] == "me"
    assert isinstance(result[1], fits.ImageHDU)
    assert result[1].data.dtype == numpy.float32
    assert result[1].name == "EXT"
    assert result[2] is table
    assert result[3] is empty


def test_promote_hdu_dtype():
    hdu = fits.PrimaryHDU(numpy.ones((2, 2), dtype="uint16"))
    assert proc.promote_hdu(hdu, totype="float64").data.dtype == numpy.float64


def test_corrector_datamodel():
    datamodel = DataModel("TEST")
    assert proc.SkyCorrector(numpy.ones(1), datamodel=datamodel).datamodel is datamodel
    assert isinstance(proc.SkyCorrector(numpy.ones(1)).datamodel, proc.SimpleDataModel)


@pytest.mark.parametrize("dtype", [">u2", "<u2", "uint8", ">i2", "int32"])
def test_corrector_promotes_integer(dtype):
    corrector = proc.BiasCorrector(numpy.full((4, 5), 0.5))
    result = corrector(image(10, dtype=dtype))
    assert result[0].data.dtype == numpy.float32
    assert numpy.allclose(result[0].data, 9.5)


def test_corrector_keeps_float():
    hdul = image(10, dtype="float64")
    result = proc.BiasCorrector(numpy.ones((4, 5)))(hdul)
    assert result[0].data.dtype == numpy.float64


def test_bpm_corrector():
    data = numpy.arange(20, dtype="float32").reshape((4, 5))
    hdul = image(0)
    hdul[0].data = data.copy()
    mask = numpy.zeros((4, 5), dtype="uint8")
    mask[1, 2] = 1

    corrector = proc.BadPixelCorrector(mask, calibid="bpmid", hwin=0, wwin=1)
    result = corrector(hdul)

    # The median of the neighbours in the row
    assert result[0].data[1, 2] == numpy.median([data[1, 1], data[1, 3]])
    good = mask == 0
    assert numpy.array_equal(result[0].data[good], data[good])
    assert result[0].header["NUM-BPM"] == "bpmid"
    assert "BPM correction with bpmid" in result[0].header["HISTORY"]


def test_bias_corrector_header():
    corrector = proc.BiasCorrector(numpy.full((4, 5), 2.0), calibid="biasid")
    assert corrector.bias_stats == 2.0
    assert not corrector.update_variance
    result = corrector(image(10))
    assert numpy.allclose(result[0].data, 8)
    history = result[0].header["HISTORY"]
    assert "Bias correction with biasid" in history
    assert "Bias image mean is 2.0" in history


def test_bias_corrector_variance():
    corrector = proc.BiasCorrector(numpy.full((4, 5), 2.0), biasvar=numpy.full((4, 5), 0.5))
    assert corrector.update_variance
    result = corrector(image(10, variance=1.0))
    assert numpy.allclose(result[0].data, 8)
    assert numpy.allclose(result["variance"].data, 1.5)


@pytest.mark.parametrize(
    "keywords, etime",
    [
        ({"DARKTIME": 3.0, "EXPTIME": 2.0}, 3.0),
        ({"EXPTIME": 2.0}, 2.0),
        ({"EXPOSED": 4.0}, 4.0),
        ({}, 1.0),
    ],
)
def test_dark_corrector_time(keywords, etime):
    corrector = proc.DarkCorrector(numpy.full((4, 5), 0.5), calibid="darkid")
    result = corrector(image(10, **keywords))
    assert numpy.allclose(result[0].data, 10 - 0.5 * etime)
    assert result[0].header["NUM-DK"] == "darkid"


def test_dark_corrector_variance():
    corrector = proc.DarkCorrector(numpy.full((4, 5), 0.5), darkvar=numpy.full((4, 5), 0.25))
    assert corrector.update_variance
    result = corrector(image(10, variance=1.0, EXPTIME=2.0))
    assert numpy.allclose(result[0].data, 9)
    assert numpy.allclose(result["variance"].data, 2.0)


def test_nonlinearity_corrector():
    corrector = proc.NonLinearityCorrector([0.5, 2.0, 1.0], calibid="linid", dtype="float64")
    result = corrector(image(3))
    # 0.5 * 3**2 + 2 * 3 + 1
    assert numpy.allclose(result[0].data, 11.5)
    assert result[0].data.dtype == numpy.float64
    assert result[0].header["NUM-LIN"] == "linid"


def test_flatfield_corrector():
    flat = numpy.full((4, 5), 2.0)
    flat[0, 0] = 4.0
    corrector = proc.FlatFieldCorrector(flat, calibid="flatid")
    assert corrector.flat_stats == flat.mean()
    result = corrector(image(8))
    expected = numpy.full((4, 5), 4.0)
    expected[0, 0] = 2.0
    assert numpy.allclose(result[0].data, expected)
    assert result[0].data.dtype == numpy.float32
    assert result[0].header["NUM-FF"] == "flatid"


def test_sky_corrector():
    corrector = proc.SkyCorrector(numpy.full((4, 5), 3.0), calibid="skyid")
    assert corrector.calib_stats == 3.0
    result = corrector(image(10))
    assert numpy.allclose(result[0].data, 7)
    assert result[0].header["NUM-SK"] == "skyid"


class NoSkyDataModel(DataModel):
    def do_sky_correction(self, img):
        return False


def test_sky_corrector_skipped():
    corrector = proc.SkyCorrector(numpy.full((4, 5), 3.0), datamodel=NoSkyDataModel("TEST"))
    result = corrector(image(10))
    assert numpy.allclose(result[0].data, 10)
    assert "NUM-SK" not in result[0].header


def test_divide_by_exposure():
    result = proc.DivideByExposure()(image(10, variance=4.0, BUNIT="ADU", EXPTIME=2.0))
    assert numpy.allclose(result[0].data, 5)
    assert numpy.allclose(result["variance"].data, 1)
    assert result[0].header["BUNIT"] == "ADU/s"
    assert result["variance"].header["BUNIT"] == "ADU2/s2"
    assert result[0].header["NUM-EXP"] == 2.0


def test_divide_by_exposure_integer():
    result = proc.DivideByExposure()(image(10, dtype="int32", BUNIT="ADU", EXPTIME=4.0))
    assert result[0].data.dtype == numpy.float32
    assert numpy.allclose(result[0].data, 2.5)


def test_divide_by_exposure_no_variance():
    result = proc.DivideByExposure()(image(10, BUNIT="adu", EXPTIME=2.0))
    assert numpy.allclose(result[0].data, 5)


@pytest.mark.parametrize("bunit", ["ADU/s", ""])
def test_divide_by_exposure_unchanged(bunit):
    result = proc.DivideByExposure()(image(10, BUNIT=bunit, EXPTIME=2.0))
    assert numpy.allclose(result[0].data, 10)
    assert "NUM-EXP" not in result[0].header


def test_divide_by_exposure_unknown_unit(caplog):
    with caplog.at_level(logging.WARNING, logger="numina.processing"):
        result = proc.DivideByExposure()(image(10, BUNIT="electron", EXPTIME=2.0))
    assert numpy.allclose(result[0].data, 10)
    assert "Unrecognized value for BUNIT electron" in caplog.text


def test_divide_by_exposure_no_unit():
    result = proc.DivideByExposure()(image(10, EXPTIME=2.0))
    assert numpy.allclose(result[0].data, 10)


def image_mask_first(value, variance, mask=7.0):
    """An image with the extension MASK before VARIANCE"""
    hdul = image(value)
    hdul.append(fits.ImageHDU(numpy.full((4, 5), mask, dtype="float32"), name="MASK"))
    hdul.append(fits.ImageHDU(numpy.full((4, 5), variance, dtype="float32"), name="VARIANCE"))
    return hdul


@pytest.mark.parametrize(
    "corrector, variance",
    [
        (proc.BiasCorrector(numpy.full((4, 5), 2.0), biasvar=numpy.full((4, 5), 0.5)), 1.5),
        (proc.DarkCorrector(numpy.full((4, 5), 0.5), darkvar=numpy.full((4, 5), 0.25)), 1.25),
    ],
)
def test_corrector_variance_not_first_extension(corrector, variance):
    """The variance is written in the extension VARIANCE, not in the second HDU (issue #270)"""
    result = corrector(image_mask_first(10, variance=1.0))
    assert numpy.allclose(result["MASK"].data, 7.0)
    assert numpy.allclose(result["VARIANCE"].data, variance)


def test_corrector_promotes_integers_to_dtype():
    """Integer images are promoted to the dtype of the corrector (issue #270)"""
    corrector = proc.BiasCorrector(numpy.full((4, 5), 2.0), dtype="float64")
    result = corrector(image(10, dtype="uint16"))
    assert result[0].data.dtype == numpy.float64
    assert numpy.allclose(result[0].data, 8)


def test_nonlinearity_corrector_logs_imgid(caplog):
    """The log shows the id of the image, not the HDUList (issue #270)"""
    corrector = proc.NonLinearityCorrector([0.0, 1.0], calibid="linid")
    with caplog.at_level(logging.DEBUG, logger="numina.processing"):
        corrector(image(3))
    assert f"correcting non linearity in {UUID}" in caplog.text
