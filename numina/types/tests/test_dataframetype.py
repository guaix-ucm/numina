
from ..frame import  DataFrameType
from ..dataframe import DataFrame


def test_dataframe_convert_none():

    datatype = DataFrameType()

    assert datatype.convert(None) is None


def test_dataframe_convert_string():

    datatype = DataFrameType()

    obj = 'filename.fits'

    result = datatype.convert(obj)

    assert isinstance(result, DataFrame)
    assert result.filename == obj
    # FIXME: no way of caomparino DataFrame for equality
    # assert result == DataFrame(filename=obj)


def test_dataframe_validate_none():

    datatype = DataFrameType()

    obj = 'filename.fits'

    assert datatype.validate(None)
