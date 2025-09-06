import pytest


def test_wheel(pattri_ins):
    dev = pattri_ins.get_device("wheel")
    assert dev.get_property("position") == 0
    dev.turn()
    assert dev.get_property("position") == 1
    assert dev.get_property("label") == "B"
    pattri_ins.get_device("wheel").position = 3
    assert dev.get_property("position") == 3
    assert dev.get_property("label") == "R"
    assert dev._capacity == 5


def test_wheel_proxy1(pattri_ins, pattri_header):
    pattri_ins.configure_with_header(pattri_header)
    assert pattri_ins.get_property("wheel.position") == 3
    assert pattri_ins.get_property("pos") == 3
    pattri_ins.pos = 4
    assert pattri_ins.get_property("wheel.position") == 4
    pattri_ins.get_device("wheel").position = 0
    assert pattri_ins.pos == 0
    pattri_ins.pos = 2
    assert pattri_ins.pos == 2
    assert pattri_ins.get_property("wheel.position") == 2


def test_wheel_proxy2(pattri_ins):
    assert pattri_ins.get_property("wheel.position") == 0
    assert pattri_ins.get_property("pos") == 0
    pattri_ins.pos = 2
    assert pattri_ins.get_property("wheel.position") == 2
    assert pattri_ins.get_device("wheel").position == 2


def test_box_proxy2(pattri_ins):
    assert pattri_ins.get_property("PSU.mode") == "Mode_A"
    assert pattri_ins.get_property("mode") == "Mode_A"
    with pytest.raises(ValueError):
        pattri_ins.mode = "R"

    with pytest.raises(ValueError):
        pattri_ins.get_device("PSU").mode = "M"
