from numina.drps.tests.pattridrp.instrument.components.wheel import PWheel


def test_wheel1(pattri_ins):
    dev = pattri_ins.get_device("wheel")
    assert isinstance(dev, PWheel)


def test_wheel2(pattri_ins):
    dev = pattri_ins.get_device("wheel")
    assert dev.position == 0
    assert dev.label == "U"


def test_wheel3(pattri_ins):
    dev = pattri_ins.get_device("wheel")
    props = dev.get_properties()
    assert "position" in props
    assert "label" in props
    # selected comes from init_config_info()
    assert "selected" in props


def test_wheel4(pattri_ins):
    dev = pattri_ins.get_device("wheel")
    props = dev.get_property_names()
    assert "position" in props
    assert "label" in props
    # selected comes from init_config_info()
    assert "selected" not in props
