"""Tests for numina.array.wavecalib.arccalibration._save_linear_wvcal_spectrum"""

from astropy.io import fits
import astropy.units as u
import numpy as np
from numpy.polynomial import Polynomial
import pytest

from numina.array.wavecalib.arccalibration import _save_linear_wvcal_spectrum

NAXIS1 = 200
# linear calibration: wavelength = 4000 + 2 * pixel (pixel from 1 to NAXIS1)
POLY_LINEAR = Polynomial([4000.0, 2.0])
# non-linear (monotonically increasing) calibration
POLY_QUADRATIC = Polynomial([4000.0, 2.0, 1e-3])


def read_spectrum(filename):
    with fits.open(filename) as hdul:
        return hdul[0].data, hdul[0].header


def wavelengths(header):
    """Wavelength at the center of each pixel (FITS convention, 1-based)."""
    pixel = np.arange(1, header["NAXIS1"] + 1)
    return header["CRVAL1"] + (pixel - header["CRPIX1"]) * header["CDELT1"]


def test_header_and_data(tmp_path):
    filename = tmp_path / "sp.fits"
    sp = np.ones(NAXIS1)
    _save_linear_wvcal_spectrum(
        sp, POLY_LINEAR, crval1=4100.0, cdelt1=0.5, naxis1_out=150, filename=filename, wv_unit=u.nm
    )
    data, header = read_spectrum(filename)
    assert header["BITPIX"] == -32  # float32
    assert data.shape == (150,)
    assert header["NAXIS1"] == 150
    assert header["CRPIX1"] == 1.0
    assert header["CRVAL1"] == 4100.0
    assert header["CDELT1"] == 0.5
    assert header["CTYPE1"] == "WAVE"
    assert header["CUNIT1"] == "nm"


@pytest.mark.parametrize("wv_unit", ["Angstrom", None, u.AA])
def test_default_unit(tmp_path, wv_unit):
    filename = tmp_path / "sp.fits"
    _save_linear_wvcal_spectrum(
        np.ones(NAXIS1), POLY_LINEAR, crval1=4002.0, cdelt1=2.0, naxis1_out=NAXIS1, filename=filename, wv_unit=wv_unit
    )
    _, header = read_spectrum(filename)
    assert header["CUNIT1"] == "Angstrom"


def test_identity_resampling(tmp_path):
    # same linear sampling as the input spectrum: the output must be
    # identical to the input
    filename = tmp_path / "sp.fits"
    rng = np.random.default_rng(1234)
    sp = rng.uniform(10, 100, NAXIS1)
    _save_linear_wvcal_spectrum(
        sp, POLY_LINEAR, crval1=POLY_LINEAR(1), cdelt1=2.0, naxis1_out=NAXIS1, filename=filename
    )
    data, _ = read_spectrum(filename)
    assert np.allclose(data, sp, rtol=1e-5)


@pytest.mark.parametrize("poly", [POLY_LINEAR, POLY_QUADRATIC])
@pytest.mark.parametrize("cdelt1", [0.3, 1.0, 3.7])
def test_flux_conservation(tmp_path, poly, cdelt1):
    # output covering the whole wavelength range of the input spectrum:
    # the total flux is preserved
    filename = tmp_path / "sp.fits"
    rng = np.random.default_rng(42)
    sp = rng.uniform(10, 100, NAXIS1)
    wvmin = poly(0.5)
    wvmax = poly(NAXIS1 + 0.5)
    crval1 = wvmin - 5 * cdelt1
    naxis1_out = int(np.ceil((wvmax - crval1) / cdelt1)) + 5
    _save_linear_wvcal_spectrum(sp, poly, crval1=crval1, cdelt1=cdelt1, naxis1_out=naxis1_out, filename=filename)
    data, _ = read_spectrum(filename)
    assert np.isclose(data.sum(), sp.sum(), rtol=1e-5)


def test_line_location(tmp_path):
    # gaussian emission line at a known wavelength, using a non-linear
    # calibration polynomial
    filename = tmp_path / "sp.fits"
    wv_line = 4250.0
    pixel = np.arange(1, NAXIS1 + 1)
    sp = 1000 * np.exp(-0.5 * ((POLY_QUADRATIC(pixel) - wv_line) / 6.0) ** 2)
    _save_linear_wvcal_spectrum(sp, POLY_QUADRATIC, crval1=4100.0, cdelt1=1.0, naxis1_out=300, filename=filename)
    data, header = read_spectrum(filename)
    wv = wavelengths(header)
    centroid = np.sum(wv * data) / np.sum(data)
    assert abs(centroid - wv_line) < 0.1
    assert abs(wv[np.argmax(data)] - wv_line) <= 1.0


def test_pixels_outside_input_range(tmp_path):
    # output pixels beyond the wavelength range of the input spectrum
    # (borders at 4001 and 4401 Angstrom) are set to zero
    filename = tmp_path / "sp.fits"
    sp = np.ones(NAXIS1)
    _save_linear_wvcal_spectrum(sp, POLY_LINEAR, crval1=3900.0, cdelt1=2.0, naxis1_out=300, filename=filename)
    data, header = read_spectrum(filename)
    wv = wavelengths(header)
    outside = (wv + 1.0 < POLY_LINEAR(0.5)) | (wv - 1.0 > POLY_LINEAR(NAXIS1 + 0.5))
    inside = (wv - 1.0 >= POLY_LINEAR(0.5)) & (wv + 1.0 <= POLY_LINEAR(NAXIS1 + 0.5))
    assert np.any(outside)
    assert np.all(data[outside] == 0)
    assert np.allclose(data[inside], 1.0, rtol=1e-5)


def test_overwrite(tmp_path):
    filename = tmp_path / "sp.fits"
    for naxis1_out in (50, 80):
        _save_linear_wvcal_spectrum(
            np.ones(NAXIS1), POLY_LINEAR, crval1=4002.0, cdelt1=2.0, naxis1_out=naxis1_out, filename=filename
        )
    data, _ = read_spectrum(filename)
    assert data.shape == (80,)


@pytest.mark.parametrize(
    "poly, cdelt1, naxis1_out, match",
    [
        (POLY_LINEAR, 0.0, 100, "CDELT1 must be > 0"),
        (POLY_LINEAR, -1.0, 100, "CDELT1 must be > 0"),
        (POLY_LINEAR, 1.0, 0, "NAXIS1 must be >= 1"),
        # decreasing wavelength for pixel > 100
        (Polynomial([4000.0, 2.0, -0.01]), 1.0, 100, "not monotonically increasing"),
    ],
)
def test_invalid_input(tmp_path, poly, cdelt1, naxis1_out, match):
    filename = tmp_path / "sp.fits"
    with pytest.raises(ValueError, match=match):
        _save_linear_wvcal_spectrum(
            np.ones(NAXIS1), poly, crval1=4002.0, cdelt1=cdelt1, naxis1_out=naxis1_out, filename=filename
        )
    assert not filename.exists()
