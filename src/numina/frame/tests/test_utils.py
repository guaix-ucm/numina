
import astropy.io.fits as fits
import numpy

from numina.frame.utils import copy_img


def test_copy_img():

    hdu1 = fits.PrimaryHDU(data=[1,2,3])

    hdr2 = fits.Header({'A': 1})
    hdu2 = fits.ImageHDU(data=[3.0,4.3,999], header=hdr2)

    hdu3 = fits.ImageHDU(data=[[1,2,3,4,5]])

    hdul = fits.HDUList([hdu1, hdu2, hdu3])

    hdul_copy = copy_img(hdul)

    assert hdul is not hdul_copy
    assert len(hdul_copy) == len(hdul_copy)
    for i in range(3):
        assert numpy.allclose(hdul[i].data, hdul_copy[i].data)
        assert hdul[i].header == hdul_copy[i].header
