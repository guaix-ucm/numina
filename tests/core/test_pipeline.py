import pytest

import numina.core.pipelineload as loader
from tests.support.drpdef import make_drp


@pytest.fixture(scope="module")
def drptest():
    return loader.drp_load("tests.support.drps", "drptest1.yaml")


def test_mode_search(drptest):

    ll = drptest.search_mode_provides("MasterBias")

    assert ll.name == "MasterBias"
    assert ll.mode == drptest.modes[ll.mode].key
    assert ll.field == "master_bias"

    ll = drptest.search_mode_provides("MasterDark")

    assert ll.name == "MasterDark"
    assert ll.mode == drptest.modes[ll.mode].key
    assert ll.field == "master_dark"


def test_mode_query(drptest):

    ll = drptest.query_provides("MasterBias")

    assert ll.name == "MasterBias"
    assert ll.mode == drptest.modes[ll.mode].key
    assert ll.field == "master_bias"

    with pytest.raises(ValueError):
        drptest.query_provides("MasterDark")

    ll = drptest.query_provides("MasterDark", search=True)

    assert ll.name == "MasterDark"
    assert ll.mode == drptest.modes[ll.mode].key
    assert ll.field == "master_dark"


DARK = "tests.support.recipes.DarkRecipe"
PIPELINES = {"default": {"dark": DARK}, "fast": {"dark": DARK}}
DRP_PIPELINES = make_drp(pipelines=PIPELINES, version="2.5")


@pytest.mark.parametrize("pipeline", ["default", "fast"])
def test_recipe_drp_headers(pipeline):
    drp = loader.drp_load_data("numina", DRP_PIPELINES)
    recipe = drp.get_recipe_object("dark", pipeline_name=pipeline)

    assert recipe.instrument == "TEST1"
    assert recipe.pipeline == pipeline
    assert recipe.drp_version == "2.5"

    hdr = recipe.set_base_headers({})
    assert hdr["NUMDRP"] == ("TEST1", "Numina DRP name")
    assert hdr["NUMDRPV"] == ("2.5", "Numina DRP version")
    assert hdr["NUMPIPE"] == (pipeline, "Numina DRP pipeline")


def test_recipe_drp_version_of_package():
    """Without version in the DRP, the version of the package"""
    import numina

    drp = loader.drp_load_data("numina", make_drp(pipelines=PIPELINES))
    recipe = drp.get_recipe_object("dark")

    assert recipe.drp_version == numina.__version__
    assert recipe.set_base_headers({})["NUMDRPV"][0] == numina.__version__


def test_recipe_state_and_mode():
    from numina.core.pipeline import ObservingMode, Pipeline

    recipes = {"dark": {"class": DARK, "kwargs": {"instrument": "OTHER"}, "state": {"version": "9.9"}}}
    pipeline = Pipeline("TEST1", "default", recipes)

    recipe = pipeline.get_recipe_object("dark")
    assert recipe.__version__ == "9.9"
    assert recipe.mode == "dark"
    # the instrument of the pipeline is used
    assert recipe.instrument == "TEST1"

    mode = ObservingMode("TEST1")
    mode.key = "dark"
    assert pipeline.load_recipe_object(mode).mode is mode


def test_products_and_provides(drptest):
    from tests.support.recipes import MasterBias

    pipeline = drptest.pipelines["default"]
    assert pipeline.load_product_class("MasterBias") is MasterBias
    assert isinstance(pipeline.load_product_from_name("MasterBias"), MasterBias)
    assert pipeline.provides("bias").name == "MasterBias"
    assert pipeline.who_provides("MasterBias").field == "master_bias"
    with pytest.raises(KeyError):
        pipeline.provides("image")


def test_search_mode_provides_not_found(drptest):
    with pytest.raises(ValueError, match="no mode provides Other"):
        drptest.search_mode_provides("Other")


def test_mode_without_recipe():
    drp = loader.drp_load_data("numina", make_drp(pipelines={"default": {"dark": DARK}, "fast": {"bias": DARK}}))
    with pytest.warns(UserWarning, match="Mode bias has not recipe"):
        with pytest.raises(ValueError):
            drp.search_mode_provides("MasterBias")


def test_product_entry():
    from numina.core.pipeline import ProductEntry

    entry = ProductEntry("drp.types.MasterBias", "bias", "master_bias")
    assert entry.alias == "MasterBias"
    assert repr(entry) == 'ProductEntry(name="drp.types.MasterBias", mode="bias", field="master_bias")'
    assert ProductEntry("MasterBias", "bias", "master_bias", alias="Bias").alias == "Bias"


@pytest.mark.parametrize(
    "keywords, expected",
    [
        ({"INSCONF": "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"}, ("225fcaf2-7f6f-49cc-972a-70fd0aee8e96", "uuid")),
        ({"INSTRUME": "TEST1"}, ("TEST1", "name")),
    ],
)
def test_select_profile_image(drptest, keywords, expected):
    import datetime

    import astropy.io.fits as fits

    hdu = fits.PrimaryHDU()
    hdu.header["DATE-OBS"] = "2020-01-02T03:04:05"
    for key, value in keywords.items():
        hdu.header[key] = value
    key, date_obs, keyname = drptest.select_profile_image(fits.HDUList([hdu]))
    assert (key, keyname) == expected
    assert date_obs == datetime.datetime(2020, 1, 2, 3, 4, 5)


class Stored:
    def __init__(self, id, content):
        self.id = id
        self.content = content


class Backend:
    def search_result_relative(self, name, result_type, partial_ob, result_desc):
        return [Stored(1, "a"), Stored(2, "b")]


def test_build_ob_result_of():
    from numina.core.oresult import ObservationResult
    from numina.core.pipeline import ObservingMode
    from numina.core.query import ResultOf

    mode = ObservingMode("TEST1")
    obsres = ObservationResult()
    assert mode.build_ob(obsres, Backend()) is obsres
    assert obsres.results == {}

    mode.build_ob_options = ResultOf("frame")
    mode.build_ob(obsres, Backend())
    assert obsres.results == {1: "a", 2: "b"}
