def test_configure1(pattri_ins, pattri_header):
    pattri_ins.configure_with_header(pattri_header)
    assert pattri_header["INSMODE"] == "Mode_B"
    assert pattri_ins.get_property("PSU.mode") == "Mode_B"


def test_configure2(pattri_ins, pattri_header):
    assert pattri_header["INSMODE"] == "Mode_B"
    pattri_ins.configure_with_header(pattri_header)
    assert pattri_ins.get_property("PSU.spaces") == []


def test_configure1b(pattri_ins, pattri_state):
    pattri_ins.configure(pattri_state)
    assert pattri_ins.get_property("PSU.mode") == "Mode_B"
    assert pattri_ins.get_property("PSU.spaces") == []
