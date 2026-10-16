"""Getters of correctors from the calibrations of a recipe"""

import astropy.io.fits as fits
import numpy

import numina.processing as proc
from numina.core import BaseRecipe, DataFrame, ObservationResult, Requirement
from numina.core.requirements import ObservationResultRequirement
from numina.datamodel import DataModel, get_imgid
from numina.processing.getters import get_corrector_for, load_corrector
from numina.types.frame import DataFrameType
from numina.util.node import IdNode

from tests.support.recipes import MasterBias

BIAS_UUID = "9a6f3c1e-2b4d-4e8f-a1c3-5d7e9f0b2c4a"


class PrefixDataModel(DataModel):
    """The identifiers of the images have a prefix, as in EMIR"""

    def get_imgid(self, img):
        return get_imgid(img, prefix=True)


class ImageRecipe(BaseRecipe):
    """Corrects the images with the mask of bad pixels and the bias, if they are given"""

    obresult = ObservationResultRequirement()
    master_bias = Requirement(MasterBias, "Master bias", optional=True)
    master_bpm = Requirement(DataFrameType, "Mask of bad pixels", optional=True)

    def create_getters(self):
        return [
            [get_corrector_for("master_bpm", proc.BadPixelCorrector, hwin=0, wwin=3)],
            [get_corrector_for("master_bias", proc.BiasCorrector)],
        ]


def image(value, uuid=None):
    hdu = fits.PrimaryHDU(numpy.full((4, 4), value, dtype="float32"))
    if uuid is not None:
        hdu.header["UUID"] = uuid
    return DataFrame(frame=fits.HDUList([hdu]))


def recipe_input(recipe, **calibrations):
    ob = ObservationResult()
    ob.frames = [image(110.0, "1c2d3e4f-0000-0000-0000-000000000000")]
    return recipe.create_input(obresult=ob, **calibrations)


def test_calibration_not_given():
    recipe = ImageRecipe()
    rinput = recipe_input(recipe)
    assert isinstance(load_corrector(rinput, "master_bias", proc.BiasCorrector, recipe.datamodel), IdNode)


def test_corrector_of_calibration():
    recipe = ImageRecipe()
    rinput = recipe_input(recipe, master_bias=image(100.0, BIAS_UUID))
    corrector = load_corrector(rinput, "master_bias", proc.BiasCorrector, recipe.datamodel)
    assert isinstance(corrector, proc.BiasCorrector)
    assert numpy.allclose(corrector.bias_stats, 100.0)
    # the identifier given by the datamodel, without prefix by default
    assert corrector.calibid == BIAS_UUID


def test_identifier_of_datamodel():
    rinput = recipe_input(ImageRecipe(), master_bias=image(100.0, BIAS_UUID))
    corrector = load_corrector(rinput, "master_bias", proc.BiasCorrector, PrefixDataModel())
    assert corrector.calibid == f"uuid:{BIAS_UUID}"


def test_arguments_of_corrector():
    rinput = recipe_input(ImageRecipe(), master_bpm=image(0.0, BIAS_UUID))
    getter = get_corrector_for("master_bpm", proc.BadPixelCorrector, hwin=0, wwin=3)
    corrector = getter(rinput, {}, None, DataModel())
    assert (corrector.hwin, corrector.wwin) == (0, 3)
    assert getter.__name__ == "get_corrector_master_bpm"


def test_flows_of_recipe():
    """The getters build the flows of the recipe, the bias is subtracted"""
    recipe = ImageRecipe()
    rinput = recipe_input(recipe, master_bias=image(100.0, BIAS_UUID))
    bpm_flow, bias_flow = recipe.init_filters(rinput, ins="unused")
    assert isinstance(bpm_flow.nodeseq[0], IdNode)
    result = bias_flow(rinput.obresult.frames[0].open())
    assert numpy.allclose(result[0].data, 10.0)
    assert result[0].header["NUM-BS"] == BIAS_UUID
