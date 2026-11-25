import copy

import astropy.io.fits as fits
import numpy
import pytest

from numina.frame.utils import copy_img


def make_hdulist():
    hdu1 = fits.PrimaryHDU(data=[1, 2, 3])

    hdr2 = fits.Header({"A": 1})
    hdu2 = fits.ImageHDU(data=[3.0, 4.3, 999], header=hdr2)

    hdu3 = fits.ImageHDU(data=[[1, 2, 3, 4, 5]])

    return fits.HDUList([hdu1, hdu2, hdu3])


def test_copy_img():
    hdul = make_hdulist()

    with pytest.warns(DeprecationWarning, match="copy.deepcopy"):
        hdul_copy = copy_img(hdul)

    assert hdul is not hdul_copy
    assert len(hdul_copy) == len(hdul)
    for i in range(3):
        assert hdul_copy[i] is not hdul[i]
        assert numpy.allclose(hdul[i].data, hdul_copy[i].data)
        assert hdul[i].header == hdul_copy[i].header


def test_deepcopy_replaces_copy_img():
    """copy.deepcopy, the replacement of copy_img, copies the data and the headers"""
    hdul = make_hdulist()
    hdul_copy = copy.deepcopy(hdul)
    hdul_copy[0].data[0] = 10
    hdul_copy[1].header["A"] = 2
    assert hdul[0].data[0] == 1
    assert hdul[1].header["A"] == 1
