"""Combination of images"""

import uuid

import astropy.io.fits as fits
import numpy
import pytest

from numina.processing.combine import combine_imgs


def create_img(value, tsutc2, extension=False):
    hdu = fits.PrimaryHDU(numpy.full((4, 4), value, dtype="float32"))
    hdu.header["TSUTC2"] = tsutc2
    hdu.header["UUID"] = str(uuid.uuid4())
    hdul = fits.HDUList([hdu])
    if extension:
        ext = fits.ImageHDU(numpy.zeros((2, 2)), name="EXTRA")
        ext.header["TSUTC2"] = tsutc2
        hdul.append(ext)
    return hdul


@pytest.mark.parametrize("extension", [False, True])
def test_combine_tsutc2_of_last_image(extension):
    imgs = [create_img(value, tsutc2, extension) for value, tsutc2 in [(1.0, 10.0), (3.0, 20.0)]]

    result = combine_imgs(imgs)

    assert result[0].header["TSUTC2"] == 20.0
    numpy.testing.assert_allclose(result[0].data, 2.0)
    assert result[0].header["NUM-NCOM"] == 2
    # the input images are not modified
    assert [img[0].header["TSUTC2"] for img in imgs] == [10.0, 20.0]
    if extension:
        assert [img["EXTRA"].header["TSUTC2"] for img in imgs] == [10.0, 20.0]
        assert result["EXTRA"].header["TSUTC2"] == 10.0


def test_combine_extensions():
    imgs = [create_img(value, 0.0, extension=True) for value in [1.0, 3.0]]

    result = combine_imgs(imgs, errors=True)

    assert [hdu.name for hdu in result] == ["PRIMARY", "EXTRA", "VARIANCE", "MAP"]
    assert result["MAP"].data.dtype == numpy.int16


def test_combine_no_images():
    with pytest.raises(ValueError):
        combine_imgs([])
