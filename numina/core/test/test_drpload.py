
import pytest

from ..pipeline import DrpSystem, InstrumentDRP
from ..pipelineload import drp_load_data

drpdata1 = """
    name: FAKE1
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


def test_drpsys_one_instrument(drpmocker):
    """Test that only one DRP is returned."""

    drpmocker.add_drp('FAKE1', drpdata1)

    drpsys = DrpSystem()

    ldrp = drpsys.query_by_name('FAKE1')

    assert isinstance(ldrp, InstrumentDRP)
    assert ldrp.name == 'FAKE1'

    ldrp2 = drpsys.query_by_name('OTHER')
    assert ldrp2 is None

    alldrps = drpsys.query_all()
    assert len(alldrps) == 1
    assert 'FAKE1' in alldrps
    # FIXME: We should check that both are equal, not just the name
    assert alldrps['FAKE1'].name == ldrp.name


def test_drpsys_faulty_instrument(drpmocker, recwarn):
    """Test that only one DRP is returned."""

    def faulty_loader():
        return 3

    drpmocker.add_drp('FAULTY', faulty_loader)

    drpsys = DrpSystem()

    ldrp = drpsys.query_by_name('FAULTY')

    assert ldrp is None

    assert len(recwarn.list) == 1

    w = recwarn.pop(RuntimeWarning)

    expected = 'Object 3 does not contain a valid DRP'
    assert str(w.message) == expected


@pytest.mark.parametrize("entryn",
                         ['other', 'fake1'],
                         ids=['different', 'lowercase']
                         )
def test_drpsys_not_valid_name(drpmocker, recwarn, entryn):

    # I like better the "with pytest.warn" mechanism
    # but I don't have yet pytest 2.8

    drpmocker.add_drp(entryn, drpdata2)

    drpsys = DrpSystem()

    ldrp = drpsys.query_by_name(entryn)

    assert ldrp is None

    assert len(recwarn.list) == 1

    w = recwarn.pop(RuntimeWarning)

    expected = 'Entry name "{}" and DRP name "{}" differ'.format(entryn, 'FAKE2')
    assert str(w.message) == expected


def test_drpsys_2_instruments(drpmocker):
    """Test that two DRPs are returned"""

    drpmocker.add_drp('FAKE1', drpdata1)
    drpmocker.add_drp('FAKE2', drpdata2)

    drpsys = DrpSystem()

    ldrp1 = drpsys.query_by_name('FAKE1')

    assert isinstance(ldrp1, InstrumentDRP)
    assert ldrp1.name == 'FAKE1'

    ldrp2 = drpsys.query_by_name('FAKE2')

    assert isinstance(ldrp2, InstrumentDRP)
    assert ldrp2.name == 'FAKE2'

    ldrp3 = drpsys.query_by_name('OTHER')
    assert ldrp3 is None

    alldrps = drpsys.query_all()
    assert len(alldrps) == 2
    assert 'FAKE1' in alldrps
    # FIXME: We should check that both are equal, not just the name
    assert alldrps['FAKE1'].name == ldrp1.name

    assert 'FAKE2' in alldrps
    # FIXME: We should check that both are equal, not just the name
    assert alldrps['FAKE2'].name == ldrp2.name


@pytest.mark.usefixtures("drpmocker")
def test_drpsys_no_instrument():

    drpsys = DrpSystem()

    ldrp1 = drpsys.query_by_name('FAKE1')

    assert ldrp1 is None

    ldrp2 = drpsys.query_by_name('FAKE2')

    assert ldrp2 is None

    ldrp3 = drpsys.query_by_name('OTHER')
    assert ldrp3 is None

    alldrps = drpsys.query_all()
    assert len(alldrps) == 0


def test_drpsys_from_cache(drpmocker):
    """Test that only one DRP is returned."""

    drpmocker.add_drp('FAKE1', drpdata1)

    drpsys = DrpSystem()

    ldrp1 = drpsys.query_by_name('FAKE1')

    assert isinstance(ldrp1, InstrumentDRP)
    assert ldrp1.name == 'FAKE1'

    ldrp2 = drpsys.query_by_name('FAKE1')

    assert isinstance(ldrp2, InstrumentDRP)
    assert ldrp2.name == 'FAKE1'

    assert ldrp1 is ldrp2


def test_ins_check_not_valid_data(recwarn):

    # I like better the "with pytest.warn" mechanism
    # but I don't have yet pytest 2.8

    drpsys = DrpSystem()

    result = drpsys.instrumentdrp_check("astring", 'noname')

    assert result is False

    assert len(recwarn.list) == 1

    w = recwarn.pop(RuntimeWarning)

    assert str(w.message) == "Object 'astring' does not contain a valid DRP"


@pytest.mark.parametrize("entryn",
                         ['other', 'fake1'],
                         ids=['different', 'lowercase']
                         )
def test_ins_check_not_valid_name(recwarn, entryn):

    # I like better the "with pytest.warn" mechanism
    # but I don't have yet pytest 2.8

    ldrp1 = drp_load_data(drpdata1)

    drpsys = DrpSystem()

    result = drpsys.instrumentdrp_check(ldrp1, entryn)

    assert result is False

    assert len(recwarn.list) == 1

    w = recwarn.pop(RuntimeWarning)

    expected = 'Entry name "{}" and DRP name "{}" differ'.format(entryn, ldrp1.name)
    assert str(w.message) == expected
