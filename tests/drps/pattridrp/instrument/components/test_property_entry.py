def test_box_prop1(pattri_ins):
    assert pattri_ins.get_property("PSU.spaces") == [{"mode_a_par": -20}]


def test_ins_prop1(pattri_ins):
    # assert pattri_ins._internal_state == {'key2': 'A'}
    assert pattri_ins.get_property("PATTRI.key20") == -1.0
    # assert pattri_ins._internal_state == {'key2': 'A'}
    pattri_ins.key2 = "B"
    # assert pattri_ins._internal_state == {'key2': 'B'}
    assert pattri_ins.get_property("PATTRI.key20") == 1


def test_ins_prop2(pattri_ins):
    # assert pattri_ins._internal_state == {'key2': 'A'}
    assert pattri_ins.key3 == "B"
    # assert pattri_ins._internal_state == {'key2': 'A', 'key3': 'B'}


def test_ins_prop3(pattri_ins):
    assert pattri_ins.get_property("PATTRI.key21") == -2.0
    assert pattri_ins.get_device("PSU")._internal_state["tag0"] == "tag0_A"
    pattri_ins.get_device("PSU").tag0 = "tag0_B"
    assert pattri_ins.get_device("PSU")._internal_state["tag0"] == "tag0_B"
    assert pattri_ins.get_property("PATTRI.key21") == 2.0
