import astropy.io.fits as fits
import numpy

from numina.types.frame import DataFrameType
from numina.types.dataframe import DataFrame


def test_dataframe_convert_none():

    datatype = DataFrameType()

    assert datatype.convert(None) is None


def test_dataframe_convert_string():

    datatype = DataFrameType()

    obj = "filename.fits"

    result = datatype.convert(obj)

    assert isinstance(result, DataFrame)
    assert result.filename == obj
    # FIXME: no way of caomparino DataFrame for equality
    # assert result == DataFrame(filename=obj)


def test_dataframe_validate_none():
    datatype = DataFrameType()
    assert datatype.validate(None)


class RecordType(DataFrameType):
    """Record the HDUList validated"""

    def validate_hdulist(self, hdulist):
        self.hdulist = hdulist


def test_dataframe_validate_file(tmp_path):
    filename = tmp_path / "frame.fits"
    fits.PrimaryHDU(numpy.zeros((2, 2))).writeto(filename)
    datatype = RecordType()
    assert datatype.validate(DataFrame(filename=str(filename)))
    # the file opened by validate is closed
    assert datatype.hdulist.fileinfo(0)["file"].closed


def test_dataframe_validate_in_memory():
    hdulist = fits.HDUList([fits.PrimaryHDU(numpy.zeros((2, 2)))])
    datatype = RecordType()
    assert datatype.validate(DataFrame(frame=hdulist))
    assert datatype.hdulist is hdulist
    assert datatype.validate(hdulist)
    assert datatype.hdulist is hdulist
    # the frame can still be used
    assert hdulist[0].data.shape == (2, 2)
