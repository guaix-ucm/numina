
import pkg_resources
import pkgutil

from numina.drps.drpsystem import DrpSystem
from numina.core.pipeline import InstrumentDRP, Pipeline


def assert_valid_instrument(instrument):
    assert isinstance(instrument, InstrumentDRP)

    pipes = instrument.pipelines
    assert 'default' in pipes
    for k, v in pipes.items():
        assert k == v.name
        assert isinstance(v, Pipeline)


def test_fake_pipeline(monkeypatch):

    def mockreturn(group=None):

        def fake_loader():
            confs = None
            modes = None
            pipelines = {'default': Pipeline('default', {}, 1)}
            fake = InstrumentDRP('FAKE', confs, modes, pipelines)
            return fake

        ep = pkg_resources.EntryPoint('fake', 'fake.loader')
        monkeypatch.setattr(ep, 'load', lambda: fake_loader)
        return [ep]

    monkeypatch.setattr(pkg_resources, 'iter_entry_points', mockreturn)

    alldrps = DrpSystem().load().query_all()
    for k, v in alldrps.items():
        assert_valid_instrument(v)


def test_fake_pipeline_alt(drpmocker):

    drpdata1 = pkgutil.get_data('numina.drps.tests', 'drptest1.yaml')

    drpmocker.add_drp('TEST1', drpdata1)

    mydrp = DrpSystem().load().query_by_name('TEST1')
    assert mydrp is not None

    assert_valid_instrument(mydrp)


def test_fake_pipeline_alt2(drpmocker):

    drpdata1 = pkgutil.get_data('numina.drps.tests', 'drptest1.yaml')

    ob_to_test = """
    id: 4
    mode: bias
    instrument: TEST1
    images:
     - ThAr_LR-U.fits
    """

    drpmocker.add_drp('TEST1', drpdata1)

    import yaml
    from numina.core.oresult import obsres_from_dict
    from numina.tests.recipes import BiasRecipe

    oblock = obsres_from_dict(yaml.safe_load(ob_to_test))

    drp = DrpSystem().load().query_by_name(oblock.instrument)

    assert drp is not None

    assert_valid_instrument(drp)

    this_pipeline = drp.pipelines[oblock.pipeline]
    expected = 'numina.tests.recipes.BiasRecipe'
    assert this_pipeline.get_recipe(oblock.mode) == expected

    recipe = this_pipeline.get_recipe_object(oblock.mode)
    assert isinstance(recipe, BiasRecipe)

    assert recipe.instrument == "TEST1"
    assert recipe.mode == "bias"
    assert recipe.simulate_error == True
