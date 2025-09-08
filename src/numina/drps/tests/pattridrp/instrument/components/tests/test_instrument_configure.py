def test_configure1(pattri_ins, pattri_state):
    pattri_ins.configure(pattri_state)
    assert pattri_ins.get_device("wheel").label == "U"
    assert pattri_ins.get_property("wheel.position") == 0
    assert pattri_ins.get_property("robot.arm_2.angle") == 1.2


def test_configure2(pattri_ins, pattri_state):
    pattri_ins.configure(pattri_state)

    res = pattri_ins.config_info()
    assert "PATTRI.wheel" in res


def test_configure3(pattri_ins, pattri_state):
    pattri_ins.configure(pattri_state)

    res = pattri_ins.config_info()
    assert "PATTRI.detector" in res
    assert pattri_ins.get_device("detector").shape == [2048, 2048]
    assert res["PATTRI.detector"] == {}


def test_configure4(pattri_ins, pattri_state):
    pattri_ins.configure(pattri_state)
    info = pattri_ins.config_info()
    assert info["PATTRI.PSU.BarL_1"] == {"pos": -8}
    assert pattri_ins.get_device("PATTRI.PSU.BarL_1").pos == -8
    assert pattri_ins.get_property("PATTRI.PSU.BarL_1.pos") == -8
