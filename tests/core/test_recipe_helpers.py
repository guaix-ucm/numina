"""Helpers of BaseRecipe: intermediate files, filters and timeit"""

import json
import logging

import astropy.io.fits as fits
import numpy
import pytest

import numina.ext.gtc
from numina.core import BaseRecipe, DataFrame, ObservationResult, Requirement, Result
from numina.core.recipes import timeit
from numina.core.requirements import ObservationResultRequirement
from numina.types.frame import DataFrameType
from numina.util.flow import SerialFlow
from numina.util.node import IdNode


class EmptyRecipe(BaseRecipe):
    pass


def test_default_run():
    recipe = EmptyRecipe()
    result = recipe.run(None)
    assert isinstance(result, EmptyRecipe.RecipeResult)
    assert recipe.create_getters() == []


@pytest.mark.parametrize("intermediate", [True, False])
def test_save_intermediate(tmp_path, intermediate):
    recipe = EmptyRecipe(intermediate_results=intermediate)
    hdul = fits.HDUList([fits.PrimaryHDU(numpy.ones((2, 2)))])

    recipe.save_intermediate_img(hdul, tmp_path / "img.fits")
    recipe.save_intermediate_array(numpy.zeros((2, 2)), tmp_path / "array.fits")
    recipe.save_structured_as_json({"a": 1}, tmp_path / "dict.json")

    assert (tmp_path / "img.fits").exists() is intermediate
    assert (tmp_path / "array.fits").exists() is intermediate
    assert (tmp_path / "dict.json").exists() is intermediate
    if intermediate:
        with fits.open(tmp_path / "array.fits") as saved:
            assert numpy.all(saved[0].data == 0)


class WithState:
    def __getstate__(self):
        return {"state": 1}


class Plain:
    def __init__(self):
        self.value = 2


@pytest.mark.parametrize(
    "obj, expected",
    [
        (WithState(), {"state": 1}),
        (Plain(), {"value": 2}),
        ({"a": 1}, {"a": 1}),
    ],
)
def test_save_structured_as_json(tmp_path, obj, expected):
    recipe = EmptyRecipe(intermediate_results=True)
    recipe.save_structured_as_json(obj, tmp_path / "obj.json")
    assert json.loads((tmp_path / "obj.json").read_text()) == expected


class FilterRecipe(BaseRecipe):
    obresult = ObservationResultRequirement()
    master_bpm = Requirement(DataFrameType, "Mask of bad pixels")


class OtherType(DataFrameType):
    pass


def getter(*args):
    return IdNode()


def test_get_filters():
    recipe = FilterRecipe()
    other = getter
    imgtypes = [None, None, DataFrameType, DataFrameType, OtherType]
    getters = [getter, [getter, getter], [other], other, getter]

    used = recipe.get_filters(imgtypes, getters)

    # the getters of OtherType are not used, no requirement has this type
    assert used == [getter, getter, getter, other, other]


class InitFiltersRecipe(FilterRecipe):
    def create_getters(self):
        return [[getter], [getter, getter]]


@pytest.fixture
def rinput():
    hdul = fits.HDUList([fits.PrimaryHDU(numpy.ones((2, 2)))])
    hdul[0].header["UUID"] = "1c9b6a4e-7d2f-4b3a-9e8c-0f5d2a7b6c31"
    obresult = ObservationResult()
    obresult.frames = [DataFrame(frame=hdul)]
    obresult.configuration = "configuration"
    recipe = InitFiltersRecipe()
    return recipe.create_input(obresult=obresult, master_bpm=DataFrame(frame=hdul))


def test_init_filters(rinput, caplog):
    recipe = InitFiltersRecipe()
    with caplog.at_level(logging.DEBUG):
        flows = recipe.init_filters(rinput)

    assert [len(flow) for flow in flows] == [1, 2]
    assert all(isinstance(flow, SerialFlow) for flow in flows)
    assert "running outside of GTC environment" in caplog.text


def test_init_filters_gtc(rinput, caplog, monkeypatch):
    monkeypatch.setattr(numina.ext.gtc, "_run_in_gtc", True)
    monkeypatch.setattr(numina.ext.gtc, "_ignore_gtc_check", False)
    recipe = InitFiltersRecipe()
    with caplog.at_level(logging.DEBUG):
        recipe.init_filters(rinput, ins="other")
    assert "running in GTC environment" in caplog.text


def test_extract_tags_from_ref():
    hdu = fits.PrimaryHDU()
    hdu.header["INSMODE"] = "LCB"
    recipe = EmptyRecipe()
    tags = recipe.extract_tags_from_ref(fits.HDUList([hdu]), ["insmode", "instrument"], base={"instrument": "I"})
    assert tags == {"insmode": "LCB", "instrument": "I"}


class TimedRecipe(BaseRecipe):
    frame = Result(DataFrameType)

    @timeit
    def run(self, rinput):
        hdul = fits.HDUList([fits.PrimaryHDU(numpy.ones((2, 2)))])
        return self.create_result(frame=hdul)


def test_timeit(caplog):
    with caplog.at_level(logging.INFO):
        result = TimedRecipe().run(None)
    header = result.frame.open()[0].header
    assert header["NUMUTC1"] <= header["NUMUTC2"]
    assert "total time measured" in caplog.text
