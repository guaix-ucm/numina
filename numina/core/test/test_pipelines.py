
from numina.core.pipeline import DrpSystem
from numina.core.pipeline import LoadableDRP, Instrument, Pipeline
from numina.core.pipelineload import drp_load_data
import pkg_resources


def assert_valid_instrument(instrument):
    assert isinstance(instrument, Instrument)

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
            fake = Instrument('FAKE', confs, modes, pipelines)
            return LoadableDRP({'fake': fake})

        ep = pkg_resources.EntryPoint('fake', 'fake.loader')
        monkeypatch.setattr(ep, 'load', lambda: fake_loader)
        return [ep]

    monkeypatch.setattr(pkg_resources, 'iter_entry_points', mockreturn)

    alldrps = DrpSystem().query_all()
    for k, v in alldrps.items():
        assert_valid_instrument(v)


def test_fake_pipeline_alt(drpmocker):

    drpdata = """
        name: FAKE
        configurations:
            default: {}
        modes:
            - description: A recipe that always fails
              key: fail
              name: Fail
              tagger:
                 - KEY1
                 - KEY2
            - description: Bias
              key: bias
              name: Bias
              tagger:
                 - KEY3
        pipelines:
            default:
                recipes:
                    bias: fake.recipes.BiasRecipe
                    fail: numina.core.utils.AlwaysFailRecipe
                version: 1
    """

    drpmocker.add_drp('FAKE', drpdata)

    mydrp = DrpSystem().query_by_name('FAKE')
    assert mydrp is not None

    assert_valid_instrument(mydrp)
    for m in mydrp.modes:
        assert m.tagger is not None


def test_fake_pipeline_alt2(drpmocker):

    drpdata = """
        name: FAKE
        configurations:
            default: {}
        modes:
            - description: A recipe that always fails
              key: fail
              name: Fail
              tagger:
                 - KEY1
                 - KEY2
            - description: Bias
              key: bias
              name: Bias
              tagger:
                 - KEY3
        pipelines:
            default:
                recipes:
                    bias: fake.recipes.BiasRecipe
                    fail: numina.core.utils.AlwaysFailRecipe
                version: 1
    """

    ob_to_test = """
    id: 4
    mode: bias
    instrument: FAKE
    images:
     - ThAr_LR-U.fits
    """

    drpmocker.add_drp('FAKE', drpdata)

    import yaml
    from numina.core.oresult import obsres_from_dict

    oblock = obsres_from_dict(yaml.load(ob_to_test))

    drp = DrpSystem().query_by_name(oblock.instrument)

    assert drp is not None

    assert_valid_instrument(drp)
    for m in drp.modes:
        assert m.tagger is not None

    assert drp.pipelines[oblock.pipeline].get_recipe(oblock.mode) == 'fake.recipes.BiasRecipe'