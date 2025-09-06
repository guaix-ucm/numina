import pytest

from ..device import DeviceBase
from ..state import Status


def test_device_base():
    dev = DeviceBase("dev1")

    assert dev.is_configured is False


def test_device_parent1():
    base = DeviceBase("base")

    other1 = DeviceBase("other1", parent=base)
    other2 = DeviceBase("other2", parent=base)

    acc1 = base.get_device("base.other1")
    acc2 = base.get_device("base.other2")

    assert acc1 is other1
    assert acc2 is other2


def test_device_parent2():

    base = DeviceBase("base")
    other1_1 = DeviceBase("other1", parent=base)  # noqa: F841
    other1_2 = DeviceBase("other1")

    # Duplicated name in tree
    with pytest.raises(ValueError):
        other1_2.set_parent(base)


def test_device_from():
    obj = DeviceBase.from_component("Base", "obj1")
    assert isinstance(obj, DeviceBase)
    assert obj.name == "obj1"
    assert obj.origin is None
    assert obj.parent is None


def test_device_set_parent():
    obj1 = DeviceBase("obj1")
    obj2 = DeviceBase("obj2", parent=obj1)
    obj1b = DeviceBase("obj1b")
    assert len(obj1b.children) == 0
    assert obj1.children["obj2"] is obj2
    assert obj2.parent is obj1
    obj2.set_parent(obj1b)
    assert obj2.parent is obj1b
    assert obj1b.children["obj2"] is obj2
    assert len(obj1.children) == 0


def test_device_get_value():
    obj1 = DeviceBase("obj1")
    u = obj1.get_value("name")
    assert u == "obj1"


def test_configure_image():
    import astropy.io.fits as fits

    test_hdu = fits.PrimaryHDU()
    test_img = fits.HDUList([test_hdu])
    obj1 = DeviceBase("obj1")
    obj2 = DeviceBase("obj2", parent=obj1)  # noqa
    obj1.configure_with_image(test_img)
    assert obj1.state.current == Status.STATUS_ACTIVE


def test_property_names():
    obj1 = DeviceBase("obj1")
    assert obj1.get_property_names() == set([])


def test_get_properties():
    obj1 = DeviceBase("obj1")
    assert obj1.get_properties(init=False) == {}


def test_end():
    obj1 = DeviceBase("obj1")
    obj2 = DeviceBase("obj2", parent=obj1)
    res1 = obj1.end_config_info({})
    assert res1 == {"children": ["obj2"]}
    res2 = obj2.end_config_info({})
    assert res2 == {}
