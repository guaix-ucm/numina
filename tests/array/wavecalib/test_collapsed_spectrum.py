"""Tests for numina.array.wavecalib.core.collapsed_spectrum"""

from astropy.io import fits
import numpy as np
import pytest

from numina.array.wavecalib.core import collapsed_spectrum


@pytest.fixture
def image2d():
    rng = np.random.default_rng(1234)
    return rng.uniform(0, 100, size=(5, 50))


@pytest.mark.parametrize("method, func", [("mean", np.mean), ("median", np.median)])
def test_2d_image(tmp_path, image2d, method, func):
    filename = tmp_path / "image2d.fits"
    fits.PrimaryHDU(image2d).writeto(filename)
    sp = collapsed_spectrum(filename, ns1=2, ns2=4, method=method)
    assert np.allclose(sp, func(image2d[1:4], axis=0))


def test_1d_image(tmp_path, image2d):
    # a 1D image (NAXIS=1) is handled as a 2D image with NAXIS2=1
    filename = tmp_path / "image1d.fits"
    fits.PrimaryHDU(image2d[0]).writeto(filename)
    sp = collapsed_spectrum(filename, ns1=1, ns2=1)
    assert sp.shape == (50,)
    assert np.allclose(sp, image2d[0])


def test_1d_image_invalid_scans(tmp_path, image2d):
    filename = tmp_path / "image1d.fits"
    fits.PrimaryHDU(image2d[0]).writeto(filename)
    with pytest.raises(SystemExit):
        collapsed_spectrum(filename, ns1=1, ns2=2)


def test_3d_image(tmp_path):
    filename = tmp_path / "image3d.fits"
    fits.PrimaryHDU(np.zeros((2, 3, 4))).writeto(filename)
    with pytest.raises(ValueError, match="NAXIS=3"):
        collapsed_spectrum(filename, ns1=1, ns2=1)


def test_no_data(tmp_path):
    filename = tmp_path / "empty.fits"
    fits.PrimaryHDU().writeto(filename)
    with pytest.raises(ValueError, match="No data found"):
        collapsed_spectrum(filename, ns1=1, ns2=1)
