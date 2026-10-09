from numina.instrument.generic import ComponentGeneric
from numina.instrument.hwdevice import HWDevice


def test_instrument0(pattri_ins):
    assert isinstance(pattri_ins, ComponentGeneric)
    assert pattri_ins.__class__.__name__ == "InstrumentGeneric_PATTRI"
    assert pattri_ins.name == "PATTRI"


def test_instrument1(pattri_ins):
    base_names = {
        "detector": "Detector_Model1",
        "PSU": "PSU",
        "robot": "RobotPos",
        "wheel": "PWheel",
    }

    for child in base_names:
        obj = pattri_ins.children[child]
        comp_id = child
        assert obj.name == comp_id

        if child in ["PSU", "detector"]:
            assert isinstance(obj, ComponentGeneric)
            assert obj.__class__.__name__ == f"ComponentGeneric_{base_names[child]}"
        if child in ["robot", "wheel"]:
            assert isinstance(obj, HWDevice)
            assert obj.__class__.__name__ == base_names[child]


def test_comp_psu_props(pattri_ins):

    assert pattri_ins.get_property("PSU.box0") == 1
    assert pattri_ins.get_property("PSU.csupos") == 444
    assert pattri_ins.get_property("PSU.box1") == 134
    assert pattri_ins.get_property("PSU.spaces") == [{"mode_a_par": -20}]
    assert pattri_ins.get_property("PSU.other") == [-1]

    psu_dev = pattri_ins.get_device("PSU")
    assert psu_dev.get_property("box0") == 1
    assert psu_dev.get_property("csupos") == 444
    assert psu_dev.get_property("box1") == 134
    assert psu_dev.get_property("spaces") == [{"mode_a_par": -20}]
    assert psu_dev.get_property("other") == [-1]


def test_comp_psu_vals(pattri_ins):
    assert pattri_ins.get_value("PSU.box0") == 1
    assert pattri_ins.get_value("PSU.csupos") == 444
    assert pattri_ins.get_value("PSU.box1", filter="V") == 7126
    assert pattri_ins.get_value("PSU.spaces", insmode="Mode_B") == []
    assert pattri_ins.get_value("PSU.other", tag0="tag0_A", tag1="tag1_Y") == [-2]


def test_comp_psu_vals2(pattri_ins):
    dev = pattri_ins.get_device("PSU")
    assert dev._internal_state == {
        "filter": "B",
        "insmode": "Mode_A",
        "tag0": "tag0_A",
        "tag1": "tag1_X",
    }


def test_comp_psu_comps(pattri_ins):
    assert pattri_ins.get_property("PSU.BarL_1.const1") == 444
    assert pattri_ins.get_property("PSU.BarL_2.const1") == 444
    assert pattri_ins.get_property("PSU.BarR_1.const1") == 444
    assert pattri_ins.get_property("PSU.BarR_2.const1") == 444


def test_comp_robo_comps(pattri_ins):
    assert pattri_ins.get_property("robot.arm_1.angle") == 0
    assert pattri_ins.get_property("robot.arm_1.angle") == 0
    assert pattri_ins.get_property("robot.arm_5.active") is True
    pattri_ins.get_device("robot.arm_4").active = False
    assert pattri_ins.get_property("robot.arm_4.active") is False
    dev = pattri_ins.get_device("robot.arm_4")
    dev.get_property("active")


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


def test_configure(pattri_ins, pattri_header2):
    pattri_ins.configure_with_header(pattri_header2)
    assert pattri_ins.get_device("wheel").position == 3
    assert pattri_ins.get_property("wheel.position") == 3
    assert pattri_ins.get_property("PSU.box1") == 803
    assert pattri_ins.get_property("PSU.spaces") == [{"mode_a_par": -20}]


def test_configure2(pattri_ins, pattri_header2):
    pattri_ins.configure_with_header(pattri_header2)
    dev = pattri_ins.get_device("PSU")
    assert dev.other == [-1]
    assert pattri_ins.get_property("PSU.other") == [-1]


def test_configure3(pattri_ins, pattri_header):
    pattri_ins.configure_with_header(pattri_header)
    assert pattri_ins.get_device("wheel").label == "R"
    assert pattri_ins.get_property("wheel.position") == 3


def test_configure4(pattri_ins, pattri_header):
    pattri_ins.configure_with_header(pattri_header)
    assert pattri_ins.depends_on() == {"filter", "insmode", "key2", "tag0", "tag1"}


def test_property5(pattri_ins):
    for u in sorted(pattri_ins._property_names):
        print(u)
    print("---")
    for u in sorted(pattri_ins._property_configurable):
        print(u)

    print("--")
    for u in pattri_ins.get_property_names():
        print(u)
    assert pattri_ins.get_property_names() == pattri_ins._property_configurable
