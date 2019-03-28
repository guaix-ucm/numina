
import pytest

from ..device import DeviceBase


def test_device_base():
    dev = DeviceBase('dev1')

    assert dev.is_configured == False


def test_device_parent1():
    base = DeviceBase('base')

    other1 = DeviceBase('other1', parent=base)
    other2 = DeviceBase('other2', parent=base)

    acc1 = base.get_device('base.other1')
    acc2 = base.get_device('base.other2')

    assert acc1 is other1
    assert acc2 is other2


def test_device_parent2():

    base = DeviceBase('base')
    other1_1 = DeviceBase('other1', parent=base)
    other1_2 = DeviceBase('other1')

    # Duplicated name in tree
    with pytest.raises(ValueError):
        other1_2.set_parent(base)
