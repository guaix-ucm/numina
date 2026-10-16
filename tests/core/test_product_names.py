"""Names of the files of the products of a result, from a template"""

import json

import astropy.io.fits as fits
import numpy
import pytest

from numina.core import Result
from numina.core.recipeinout import RecipeResult
from numina.dal.utils import PRODUCT_TEMPLATE_FIELDS, check_template
from numina.types.array import ArrayType
from numina.types.dataframe import DataFrame
from numina.types.datatype import ListOfType
from numina.types.frame import DataFrameType
from numina.types.structured import BaseStructuredCalibration

FRAME_UUID = "11111111-2222-3333-4444-555555555555"


class ResultWithProducts(RecipeResult):
    image = Result(DataFrameType)
    images = Result(ListOfType(DataFrameType))
    nouuid = Result(DataFrameType)
    calib = Result(BaseStructuredCalibration)
    values = Result(ArrayType)


def frame(uuid=None):
    hdul = fits.HDUList([fits.PrimaryHDU()])
    if uuid is not None:
        hdul[0].header["UUID"] = uuid
    return DataFrame(frame=hdul)


@pytest.fixture
def result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return ResultWithProducts(
        image=frame(FRAME_UUID),
        images=[frame("aaaaaaaa-0000-0000-0000-000000000000"), frame("bbbbbbbb-0000-0000-0000-000000000000")],
        nouuid=frame(),
        calib=BaseStructuredCalibration(instrument="TEST1"),
        values=numpy.array([1.0, 2.0]),
    )


def test_default_names(result, tmp_path):
    """Without template, the names of the fields, as before"""
    values = result.store_to(None)["values"]
    assert values == {
        "image": "image.fits",
        "images": ["images0.fits", "images1.fits"],
        "nouuid": "nouuid.fits",
        "calib": "calib.json",
        "values": "values.txt",
    }
    for name in ["image.fits", "images0.fits", "nouuid.fits", "calib.json", "values.txt"]:
        assert (tmp_path / name).is_file()


def test_names_with_uuid(result, tmp_path):
    """Each product with its uuid, or the uuid of the result if it has none"""
    values = result.store_to(None, product_tmpl="{name}_{uuid}")["values"]
    assert values["image"] == f"image_{FRAME_UUID}.fits"
    assert values["images"] == [
        "images0_aaaaaaaa-0000-0000-0000-000000000000.fits",
        "images1_bbbbbbbb-0000-0000-0000-000000000000.fits",
    ]
    assert values["nouuid"] == f"nouuid_{result.uuid}.fits"
    assert values["calib"] == f"calib_{result.calib.uuid}.json"
    assert values["values"] == f"values_{result.uuid}.txt"
    for value in values.values():
        for name in value if isinstance(value, list) else [value]:
            assert (tmp_path / name).is_file()
    # the uuid in the name is that of the content
    assert fits.getheader(tmp_path / values["image"])["UUID"] == FRAME_UUID
    assert json.loads((tmp_path / values["calib"]).read_text())["uuid"] == result.calib.uuid


def test_names_with_other_fields(result):
    fields = {"obsid": "bias1", "taskid": 7}
    values = result.store_to(None, product_tmpl="{obsid}_{taskid}_{name}", fields=fields)["values"]
    assert values["image"] == "bias1_7_image.fits"
    assert values["images"] == ["bias1_7_images0.fits", "bias1_7_images1.fits"]


class ResultWithOptional(RecipeResult):
    image = Result(DataFrameType)
    other = Result(DataFrameType, optional=True)


def test_optional_product_not_stored(tmp_path, monkeypatch):
    """A product with value None is not saved, as without template"""
    monkeypatch.chdir(tmp_path)
    result = ResultWithOptional(image=frame(FRAME_UUID))
    values = result.store_to(None, product_tmpl="{name}_{uuid}")["values"]
    assert values == {"image": f"image_{FRAME_UUID}.fits", "other": None}


@pytest.mark.parametrize("template", ["{name}_{uuid}", "{obsid}_{taskid}_{name}"])
def test_valid_product_templates(template):
    assert check_template(template, PRODUCT_TEMPLATE_FIELDS) == template


def test_invalid_product_template():
    with pytest.raises(ValueError, match="field '{date}' not allowed in template"):
        check_template("{name}_{date}", PRODUCT_TEMPLATE_FIELDS)
