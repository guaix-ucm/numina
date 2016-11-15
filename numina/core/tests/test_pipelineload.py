
import pytest

from ..pipelineload import drp_load
from ..pipeline import ObservingMode


@pytest.fixture(scope='module')
def insdrp():
    return drp_load('numina.drps.tests', 'drptest1.yaml')


def test_drp_product_query(insdrp):

    assert insdrp is not None

    assert len(insdrp.products) == 1

    result = insdrp.query_mode_provides('MasterBias')

    assert len(result) == 2
    assert isinstance(result[0], ObservingMode)
    assert result[0].key == 'bias'
    assert result[1] == 'master_bias'


def test_drp_product_query_fail(insdrp):

    with pytest.raises(ValueError):
        insdrp.query_mode_provides('MasterFull')

