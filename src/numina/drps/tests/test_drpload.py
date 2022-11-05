
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

    with pytest.raises(KeyError):
        drpsys.query_by_name('OTHER')

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

    with pytest.raises(KeyError):
        drpsys.query_by_name('OTHER')

    alldrps = drpsys.query_all()
    assert len(alldrps) == 2
    assert 'TEST1' in alldrps
    # FIXME: We should check that both are equal, not just the name
    assert alldrps['TEST1'].name == ldrp1.name

    assert 'TEST2' in alldrps
    # FIXME: We should check that both are equal, not just the name
    assert alldrps['TEST2'].name == ldrp2.name


def test_drpsys_name_2_instruments(drpmocker):
    """Test that two DRPs are returned"""

    drpdata1 = pkgutil.get_data('numina.drps.tests', 'drptest1.yaml')
    drpdata2 = pkgutil.get_data('numina.drps.tests', 'drptest2.yaml')
    drpmocker.add_drp('TEST1', drpdata1)
    drpmocker.add_drp('TEST2', drpdata2)

    ldrp1 = DrpSystem.load_drp('TEST1')

    assert ldrp1 is not None
    assert ldrp1.name == 'TEST1'

    ldrp2 = DrpSystem.load_drp('TEST2')

    assert ldrp2 is not None
    assert ldrp2.name == 'TEST2'

    with pytest.raises(KeyError):
        DrpSystem.load_drp('OTHER')


def test_drpsys_iload_2_instruments(drpmocker):
    """Test that two DRPs are returned"""

    drpdata1 = pkgutil.get_data('numina.drps.tests', 'drptest1.yaml')
    drpdata2 = pkgutil.get_data('numina.drps.tests', 'drptest2.yaml')
    drpmocker.add_drp('TEST1', drpdata1)
    drpmocker.add_drp('TEST2', drpdata2)

    gendrp = DrpSystem.iload()

    for ldrp in gendrp:
        assert ldrp is not None


@pytest.mark.usefixtures("drpmocker")
def test_drpsys_no_instrument():

    drpsys = DrpSystem()
    drpsys.load()

    with pytest.raises(KeyError):
        drpsys.query_by_name('TEST1')

    with pytest.raises(KeyError):
        drpsys.query_by_name('TEST2')

    with pytest.raises(KeyError):
        drpsys.query_by_name('OTHER')

    alldrps = drpsys.query_all()
    assert len(alldrps) == 0


def test_drpsys_bad_file(capsys, drpmocker):
    """Test that a bad file doesn't break the load"""

    drpdata3 = pkgutil.get_data('numina.drps.tests', 'drptest3.yaml')

    drpmocker.add_drp('TEST3', drpdata3)
    drpsys = DrpSystem()
    drpsys.load()
    expected_msg = [
        '',
        "Error is:  Missing key 'modes' inside 'root' node",
        'Problem loading TEST3 = TEST3.loader'
    ]

    out, err = capsys.readouterr()
    err = err.split("\n")
    err.sort()

    assert err == expected_msg

