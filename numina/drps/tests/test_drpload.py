
import pytest
import pkgutil

from numina.drps.drpsystem import DrpSystem


def test_drpsys_one_instrument(drpmocker):
    """Test that only one DRP is returned."""

    drpdata1 = pkgutil.get_data('numina.drps.tests', 'drptest1.yaml')

    drpmocker.add_drp('TEST1', drpdata1)

    drpsys = DrpSystem()
    drpsys.load()

    ldrp = drpsys.query_by_name('TEST1')

    assert ldrp is not None
    assert ldrp.name == 'TEST1'

    res = drpsys.query_by_name('OTHER')
    assert res is None

    alldrps = drpsys.query_all()
    assert len(alldrps) == 1
    assert 'TEST1' in alldrps
    # FIXME: We should check that both are equal, not just the name
    assert alldrps['TEST1'].name == ldrp.name


def test_drpsys_2_instruments(drpmocker):
    """Test that two DRPs are returned"""

    drpdata1 = pkgutil.get_data('numina.drps.tests', 'drptest1.yaml')
    drpdata2 = pkgutil.get_data('numina.drps.tests', 'drptest2.yaml')
    drpmocker.add_drp('TEST1', drpdata1)
    drpmocker.add_drp('TEST2', drpdata2)

    drpsys = DrpSystem()
    drpsys.load()

    ldrp1 = drpsys.query_by_name('TEST1')

    assert ldrp1 is not None
    assert ldrp1.name == 'TEST1'

    ldrp2 = drpsys.query_by_name('TEST2')

    assert ldrp2 is not None
    assert ldrp2.name == 'TEST2'

    res = drpsys.query_by_name('OTHER')
    assert res is None

    alldrps = drpsys.query_all()
    assert len(alldrps) == 2
    assert 'TEST1' in alldrps
    # FIXME: We should check that both are equal, not just the name
    assert alldrps['TEST1'].name == ldrp1.name

    assert 'TEST2' in alldrps
    # FIXME: We should check that both are equal, not just the name
    assert alldrps['TEST2'].name == ldrp2.name


@pytest.mark.usefixtures("drpmocker")
def test_drpsys_no_instrument():

    drpsys = DrpSystem()
    drpsys.load()

    res = drpsys.query_by_name('TEST1')
    assert res is None

    res = drpsys.query_by_name('TEST2')
    assert res is None

    res = drpsys.query_by_name('OTHER')
    assert res is None

    alldrps = drpsys.query_all()
    assert len(alldrps) == 0
