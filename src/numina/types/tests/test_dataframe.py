"""Construction of DataFrame"""

import astropy.io.fits as fits
import numpy
import pytest

from ..dataframe import DataFrame


@pytest.fixture
def fitsfile(tmp_path):
    filename = tmp_path / "image.fits"
    fits.PrimaryHDU(data=numpy.ones((3, 3))).writeto(filename)
    return filename


def test_filename(fitsfile):
    frame = DataFrame(filename=str(fitsfile))
    with frame.open() as hdul:
        assert hdul[0].data.sum() == 9


def test_hdulist_in_memory():
    hdul = fits.HDUList([fits.PrimaryHDU(data=numpy.ones((3, 3)))])
    frame = DataFrame(frame=hdul)
    assert frame.open() is hdul


def test_hdulist_opened_from_file_warns(fitsfile):
    """numina can close it, so the data would not be readable"""
    with fits.open(fitsfile) as hdul:
        with pytest.warns(RuntimeWarning, match="HDUList opened from .*image.fits.*use DataFrame\\(filename=...\\)"):
            frame = DataFrame(frame=hdul)
        assert frame.frame is hdul


def test_empty_hdulist():
    frame = DataFrame(frame=fits.HDUList())
    assert len(frame.frame) == 0


def test_no_frame_no_filename():
    with pytest.raises(ValueError):
        DataFrame()
