"""Quality control of the data products, numina.types.product"""

import astropy.io.fits as fits
import numpy
import pytest

from numina.types.dataframe import DataFrame
from numina.types.frame import DataFrameType
from numina.types.product import DataProductMixin, DataProductType
from numina.types.qc import QC


class FrameProduct(DataProductMixin, DataFrameType):
    pass


def image(qc=None):
    hdu = fits.PrimaryHDU(numpy.zeros((2, 2)))
    hdu.header["INSTRUME"] = "TEST1"
    if qc is not None:
        hdu.header["NUMRQC"] = qc
    return fits.HDUList([hdu])


@pytest.mark.parametrize("convert", [lambda hdul: hdul, lambda hdul: DataFrame(frame=hdul)])
@pytest.mark.parametrize("qc, expected", [("GOOD", QC.GOOD), ("BAD", QC.BAD), (None, QC.UNKNOWN)])
def test_extract_db_info_frame(convert, qc, expected):
    info = FrameProduct().extract_db_info(convert(image(qc)), ["instrument"])
    assert info["quality_control"] == expected
    assert info["instrument"] == "TEST1"


@pytest.mark.parametrize(
    "obj, expected",
    [({"quality_control": QC.PARTIAL}, QC.PARTIAL), ({}, QC.UNKNOWN), (1, QC.UNKNOWN)],
)
def test_extract_db_info_other(obj, expected):
    info = DataProductType(dict).extract_db_info(obj, [])
    assert info["quality_control"] == expected


def test_state():
    product = DataProductType(dict)
    assert product.quality_control == QC.UNKNOWN
    assert product.update_meta_info()["quality_control"] == "UNKNOWN"

    product.__setstate__({"quality_control": "GOOD"})
    assert product.quality_control == QC.GOOD
    assert product.__getstate__()["quality_control"] == QC.GOOD


def test_isproduct():
    assert DataProductType.isproduct()
    assert FrameProduct().name() == "FrameProduct"
