import pytest


def test_mod_proxy1(pattri_ins, pattri_header):
    pattri_ins.configure_with_header(pattri_header)
    assert pattri_ins.get_property("key2") == "A"


def test_mod_proxy2(pattri_ins, pattri_header):
    pattri_ins.configure_with_header(pattri_header)
    assert pattri_ins.get_property("key2") == "A"
    pattri_ins.key2 = "B"
    assert pattri_ins.get_property("key2") == "B"


def test_mod_proxy3(pattri_ins, pattri_header):
    pattri_ins.configure_with_header(pattri_header)
    assert pattri_ins.get_property("key3") == "B"


def test_mod_limit1(pattri_ins):
    assert pattri_ins.get_property("key1") == 0
    assert pattri_ins.get_property("key11") == 23
    assert pattri_ins.get_property("key12") == 12
    assert pattri_ins.get_property("key13") == 8
    assert pattri_ins.get_property("key14") == 0


def test_mod_limit2(pattri_ins):
    pattri_ins.key1 = 17
    assert pattri_ins.get_property("key1") == 17
    pattri_ins.key11 = 22
    assert pattri_ins.get_property("key11") == 22


def test_mod_limit3(pattri_ins):
    with pytest.raises(ValueError):
        pattri_ins.key11 = 50
