
from numina.core import init_drp_system
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

    m = init_drp_system()
    for k, v in m.items():
        assert_valid_instrument(v)


def test_fake_pipeline_alt(monkeypatch):

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


    def mockreturn(group=None):
        ep = pkg_resources.EntryPoint('fake', 'fake.loader')

        def fake_loader():
            return drp_load_data(drpdata)

        monkeypatch.setattr(ep, 'load', lambda: fake_loader)
        return [ep]

    monkeypatch.setattr(pkg_resources, 'iter_entry_points', mockreturn)

    m = init_drp_system()
    for k, v in m.items():
        assert_valid_instrument(v)
        for m in v.modes:
            assert m.tagger is not None