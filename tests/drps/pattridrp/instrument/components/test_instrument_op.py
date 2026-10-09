def test_ini1(pattri_ins):
    pattri_ins.mode = "Mode_B"
    assert pattri_ins.get_property("mode") == "Mode_B"
    assert pattri_ins.get_property("pos") == 0
    assert pattri_ins.get_property("filter") == "U"
    assert pattri_ins.get_property("key1") == 0
    assert pattri_ins.get_property("key11") == 23
    assert pattri_ins.get_property("key12") == 12
    assert pattri_ins.get_property("key13") == 8
    assert pattri_ins.get_property("key14") == 0
    assert pattri_ins.get_property("key2") == "A"
    assert pattri_ins.get_property("key3") == "B"
    assert pattri_ins.get_property("tag0") == "tag0_A"
    assert pattri_ins.get_property("tag1") == "tag1_X"

    assert pattri_ins.get_property("PSU.csupos") == 444
    assert pattri_ins.get_property("PSU.box0") == 1
    # FIXME: is different to the value of filter
    assert pattri_ins.get_property("PSU.box1") == 134
    assert pattri_ins.get_property("PSU.tag0") == "tag0_A"
    assert pattri_ins.get_property("PSU.tag1") == "tag1_X"
    assert pattri_ins.get_property("PSU.mode") == "Mode_B"
    # This should be
    assert pattri_ins.get_property("PSU.spaces") == []
    assert pattri_ins.get_property("PSU.other") == [-1]


def test_ini2(pattri_ins):
    pattri_ins.mode = "Mode_B"
    pattri_ins.filter = "I"
    assert pattri_ins.get_property("PSU.box1") == -300


def test_ini3(pattri_ins):
    pattri_ins.mode = "Mode_B"
    pattri_ins.filter = "I"
    assert pattri_ins.get_property("PSU.spaces") == []


def test_depends(pattri_ins):
    assert pattri_ins.depends_on() == {"filter", "insmode", "key2", "tag0", "tag1"}
    assert pattri_ins.get_device("wheel").depends_on() == {"filter"}
    assert pattri_ins.get_device("PSU").depends_on() == {
        "filter",
        "insmode",
        "tag0",
        "tag1",
    }
    assert pattri_ins.get_device("robot").depends_on() == set()
    assert pattri_ins.get_device("detector").depends_on() == set()
