"""Construction of DataFrame"""

import astropy.io.fits as fits
import numpy
import pytest

from numina.types.dataframe import DataFrame


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


def test_frame_and_filename_none():
    with pytest.raises(ValueError, match="cannot be both None"):
        DataFrame()


def test_dump_dataframe_filename(fitsfile, tmp_path, monkeypatch):
    """A DataFrame with only a file is copied to the results (issue #271)"""
    from numina.types.frame import dump_dataframe

    monkeypatch.chdir(tmp_path)
    fname = dump_dataframe(DataFrame(filename=str(fitsfile)), "result_image")
    assert fname == "result_image.fits"
    with fits.open(fname) as hdul:
        assert hdul[0].data.sum() == 9


def test_dump_dataframe_frame(tmp_path, monkeypatch):
    from numina.types.frame import dump_dataframe

    monkeypatch.chdir(tmp_path)
    hdul = fits.HDUList([fits.PrimaryHDU(data=numpy.ones((2, 2)))])
    fname = dump_dataframe(DataFrame(frame=hdul), "result_image")
    assert fname == "result_image.fits"
    with fits.open(fname) as hdul2:
        assert hdul2[0].data.sum() == 4


def test_dump_dataframe_same_file(tmp_path, monkeypatch):
    """The file is not copied onto itself"""
    from numina.types.frame import dump_dataframe

    monkeypatch.chdir(tmp_path)
    fits.PrimaryHDU(data=numpy.ones((2, 2))).writeto("result_image.fits")
    fname = dump_dataframe(DataFrame(filename="result_image.fits"), "result_image")
    assert fname == "result_image.fits"
    with fits.open(fname) as hdul:
        assert hdul[0].data.sum() == 4


def test_dump_dataframe_none():
    from numina.types.frame import dump_dataframe

    assert dump_dataframe(None, "result_image") is None
