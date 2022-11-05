
import pytest

import numina.core.pipelineload as loader

from numina.tests.recipes import MasterBias, MasterDark

@pytest.fixture(scope='module')
def drptest():
    return loader.drp_load('numina.drps.tests', 'drptest1.yaml')


def test_mode_search(drptest):

    ll = drptest.search_mode_provides("MasterBias")

    assert ll.name == 'MasterBias'
    assert ll.mode == drptest.modes[ll.mode].key
    assert ll.field == 'master_bias'

    ll = drptest.search_mode_provides("MasterDark")

    assert ll.name == 'MasterDark'
    assert ll.mode == drptest.modes[ll.mode].key
    assert ll.field == 'master_dark'


def test_mode_query(drptest):

    ll = drptest.query_provides("MasterBias")

    assert ll.name == 'MasterBias'
    assert ll.mode == drptest.modes[ll.mode].key
    assert ll.field == 'master_bias'

    with pytest.raises(ValueError):
        drptest.query_provides("MasterDark")

    ll = drptest.query_provides("MasterDark", search=True)

    assert ll.name == 'MasterDark'
    assert ll.mode == drptest.modes[ll.mode].key
    assert ll.field == 'master_dark'