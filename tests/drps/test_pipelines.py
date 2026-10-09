import pkgutil

from numina.drps.drpsystem import DrpSystem
from numina.core.pipeline import InstrumentDRP, Pipeline


def assert_valid_instrument(instrument):
    assert isinstance(instrument, InstrumentDRP)

    pipes = instrument.pipelines
    assert "default" in pipes
    for k, v in pipes.items():
        assert k == v.name
        assert isinstance(v, Pipeline)


def test_fake_pipeline(drpmocker):

    def fake_loader():
        confs = dict()
        modes = dict()
        pipelines = {"default": Pipeline("FAKE", "default", {})}
        return InstrumentDRP("FAKE", confs, modes, pipelines)

    drpmocker.add_drp("FAKE", fake_loader)

    alldrps = DrpSystem().load().query_all()
    assert list(alldrps) == ["FAKE"]
    for k, v in alldrps.items():
        assert_valid_instrument(v)


def test_fake_pipeline_alt(drpmocker):

    drpdata1 = pkgutil.get_data("numina.testing.drps", "drptest1.yaml")

    drpmocker.add_drp("TEST1", drpdata1)

    mydrp = DrpSystem().load().query_by_name("TEST1")
    assert mydrp is not None

    assert_valid_instrument(mydrp)


def test_fake_pipeline_alt2(drpmocker):

    drpdata1 = pkgutil.get_data("numina.testing.drps", "drptest1.yaml")

    ob_to_test = """
    id: 4
    mode: bias
    instrument: TEST1
    images:
     - ThAr_LR-U.fits
    """

    drpmocker.add_drp("TEST1", drpdata1)

    import yaml
    from numina.core.oresult import oblock_from_dict
    from numina.testing.recipes import BiasRecipe

    oblock = oblock_from_dict(yaml.safe_load(ob_to_test))

    drp = DrpSystem().load().query_by_name(oblock.instrument)

    assert drp is not None

    assert_valid_instrument(drp)

    this_pipeline = drp.pipelines[oblock.pipeline]
    expected = "numina.testing.recipes.BiasRecipe"
    assert this_pipeline.recipes[oblock.mode]["class"] == expected

    recipe = this_pipeline.get_recipe_object(oblock.mode)
    assert isinstance(recipe, BiasRecipe)

    assert recipe.instrument == "TEST1"
    assert recipe.mode == "bias"
    assert recipe.simulate_error is True
