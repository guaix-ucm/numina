
import pkg_resources

import pytest

from ..cli import main
from numina.core.pipelineload import drp_load_data

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

drpdata2 = """
    name: FAKE2
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


def create_mock_entry_point(monkeypatch, entry_name, drpdata):

    loader = "%s.loader" % entry_name

    ep = pkg_resources.EntryPoint(entry_name, loader)

    def fake_loader():
        return drp_load_data(drpdata)

    monkeypatch.setattr(ep, 'load', lambda: fake_loader)

    return ep


class DRPMocker(object):
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self._eps = []

        def mockreturn(group=None):
            return self._eps

        self.monkeypatch.setattr(pkg_resources, 'iter_entry_points', mockreturn)

    def add_drp(self, name, data):
        ep = create_mock_entry_point(self.monkeypatch, name, data)
        self._eps.append(ep)


@pytest.fixture
def drpmocker(monkeypatch):
    """A fixture that mocks the loading of DRPs"""
    return DRPMocker(monkeypatch)


def test_show_instrument(capsys, drpmocker):
    """Test that one instrumenst is shown"""
    drpmocker.add_drp('fake', drpdata)

    expected = ("Instrument: FAKE\n"
                " has configuration 'default'\n"
                " has pipeline 'default', version 1\n"
                )

    main(['show-instruments'])

    out, err = capsys.readouterr()
    assert out == expected


@pytest.mark.xfail(reason="instruments can be output in any order")
def test_show_2_instruments(capsys, drpmocker):
    """Test that two instruments are shown"""

    # FIXME: probably instruments can be output in any order
    drpmocker.add_drp('fake', drpdata)
    drpmocker.add_drp('fake2', drpdata2)

    expected = ("Instrument: FAKE2\n"
                " has configuration 'default'\n"
                " has pipeline 'default', version 1\n"
                "Instrument: FAKE\n"
                " has configuration 'default'\n"
                " has pipeline 'default', version 1\n"
                )

    main(['show-instruments'])

    out, err = capsys.readouterr()
    assert out == expected


@pytest.mark.usefixtures("drpmocker")
def test_show_no_instrument(capsys):
    """Test that no instruments are shown"""
    expected = ""

    main(['show-instruments'])

    out, err = capsys.readouterr()
    assert out == expected
