"""Resize the images of numina.frame"""

import astropy.io.fits as fits
import numpy
import pytest

import numina.frame as nf


def test_get_hdu_shape():
    hdu = fits.PrimaryHDU(numpy.zeros((3, 4, 5)))
    assert nf.get_hdu_shape(hdu.header) == (5, 4, 3)


def test_custom_region_to_str():
    assert nf.custom_slice_to_str(slice(1, 4)) == "1:4"
    assert nf.custom_slice_to_str(slice(1, 9, 2)) == "1:9:2"
    assert nf.custom_region_to_str((slice(0, 10), slice(5, 7, 1))) == "[0:10,5:7:1]"


def test_resize_hdu():
    hdu = fits.PrimaryHDU(numpy.ones((2, 3), dtype="float32"))
    hdu.header["OBSERVER"] = "me"
    region = (slice(1, 3), slice(2, 5))

    newhdu = nf.resize_hdu(hdu, (4, 6), region, fill=-1.0)

    assert isinstance(newhdu, fits.PrimaryHDU)
    assert newhdu.data.shape == (4, 6)
    assert numpy.all(newhdu.data[region] == 1)
    assert numpy.sum(newhdu.data == -1) == 24 - 6
    assert newhdu.header["OBSERVER"] == "me"
    assert newhdu.header["NVALREGI"] == "[1:3,2:5]"
    assert "OVALREGI" not in newhdu.header


def test_resize_hdu_window():
    hdu = fits.PrimaryHDU(numpy.arange(12, dtype="float64").reshape((3, 4)))
    window = (slice(1, 3), slice(0, 2))
    region = (slice(0, 2), slice(1, 3))

    newhdu = nf.resize_hdu(hdu, (3, 3), region, window=window, dtype="float32")

    assert newhdu.data.dtype == numpy.float32
    assert numpy.array_equal(newhdu.data[region], hdu.data[window])
    assert newhdu.header["OVALREGI"] == "[1:3,0:2]"


def test_resize_hdu_input_unchanged():
    hdu = fits.PrimaryHDU(numpy.ones((2, 3)))
    nf.resize_hdu(hdu, (4, 6), (slice(1, 3), slice(2, 5)), window=(slice(0, 2), slice(0, 3)))
    assert "NVALREGI" not in hdu.header
    assert "OVALREGI" not in hdu.header


def hdulist():
    return fits.HDUList(
        [
            fits.PrimaryHDU(numpy.ones((2, 3))),
            fits.ImageHDU(numpy.full((2, 3), 2.0), name="VARIANCE"),
        ]
    )


def test_resize_hdul():
    hdul = hdulist()
    region = (slice(1, 3), slice(2, 5))

    result = nf.resize_hdul(hdul, (4, 6), region)

    assert result[0].data.shape == (4, 6)
    assert result[1] is hdul[1]


def test_resize_hdul_extensions(tmp_path):
    region = (slice(1, 3), slice(2, 5))

    result = nf.resize_hdul(hdulist(), (4, 6), region, extensions=[0, 1], fill=numpy.nan)

    assert result["VARIANCE"].data.shape == (4, 6)
    assert numpy.all(result["VARIANCE"].data[region] == 2.0)
    result.writeto(tmp_path / "resized.fits")


@pytest.mark.parametrize("as_path", [True, False])
def test_resize_fits(tmp_path, as_path):
    filename = tmp_path / "image.fits"
    newfilename = tmp_path / "resized.fits"
    hdulist().writeto(filename)
    region = (slice(1, 3), slice(2, 5))

    if as_path:
        nf.resize_fits(str(filename), newfilename, (4, 6), region, scale=1)
    else:
        with fits.open(filename) as hdul:
            nf.resize_fits(hdul, newfilename, (4, 6), region)
            assert hdul[0].data is not None

    with fits.open(newfilename) as hdul:
        assert len(hdul) == 1
        assert hdul[0].data.shape == (4, 6)
        assert numpy.all(hdul[0].data[region] == 1)


def test_resize_fits_overwrite(tmp_path):
    filename = tmp_path / "image.fits"
    newfilename = tmp_path / "resized.fits"
    hdulist().writeto(filename)
    newfilename.write_text("old")
    region = (slice(1, 3), slice(2, 5))

    with pytest.raises(OSError):
        nf.resize_fits(str(filename), newfilename, (4, 6), region, overwrite=False)

    nf.resize_fits(str(filename), newfilename, (4, 6), region)
    with fits.open(newfilename) as hdul:
        assert hdul[0].data.shape == (4, 6)
