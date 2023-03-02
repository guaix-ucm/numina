import pytest

from numina.instrument.assembly import load_paths_store, assembly_instrument
import numina.instrument.generic as comps


def load_from_tests(name, date_str):
    pkg_paths = ['numina.drps.tests.configs']
    comp_store = load_paths_store(pkg_paths)
    ins = assembly_instrument(comp_store, name, date_str)
    return ins


@pytest.fixture
def pattri_ins():
    ins = load_from_tests('PATTRI', "2016-06-01T12:00:00.0")
    return ins


def test_ins_load(pattri_ins):
    assert isinstance(pattri_ins, comps.InstrumentGeneric)


def test_ins_load_fail():

    with pytest.raises(ValueError):
        load_from_tests('PATTRI', "2006-06-01T12:00:00.0")

    with pytest.raises(ValueError):
        load_from_tests('WRONG_NAME', "2016-06-01T12:00:00.0")


def test_ins_prefix(pattri_ins):

    devpath1 = pattri_ins.get_device('PATTRI.PSU')
    devpath2 = pattri_ins.get_device('PSU')

    assert isinstance(devpath1, comps.ComponentGeneric)

    assert devpath1 is devpath2


def test_all_comps(pattri_ins):

    id_names = ['PSU', 'detector', 'wheel']
    comp_names = ['PSU', 'Detector_Model1', 'Wheel_Model1']
    comp_uuids = [
        '41fa8884-2ea3-42c3-98a6-a17a29c990cf',
        '510efa03-c266-4202-8a4d-d5799af49d9d',
        '27614307-783f-4c1c-bd9c-165ff9cec1c5'
    ]
    for id_n, c_name, c_uuid in zip(id_names, comp_names, comp_uuids):
        dev = pattri_ins.get_device(id_n)
        assert dev.origin.name == c_name
        assert str(dev.origin.uuid) == c_uuid
        assert isinstance(dev, comps.ComponentGeneric)


def test_comp_psu_props(pattri_ins):

    assert pattri_ins.get_property('PSU.box0') == 1
    assert pattri_ins.get_property('PSU.csupos') == 444
    assert pattri_ins.get_property('PSU.box1') == 453
    assert pattri_ins.get_property('PSU.spaces') == [{'mode_a_par': -20}]
    assert pattri_ins.get_property('PSU.other') == [-1]

    psu_dev = pattri_ins.get_device('PSU')
    assert psu_dev.get_property('box0') == 1
    assert psu_dev.get_property('csupos') == 444
    assert psu_dev.get_property('box1') == 453
    assert psu_dev.get_property('spaces') == [{'mode_a_par': -20}]
    assert psu_dev.get_property('other') == [-1]


def test_comp_psu_vals(pattri_ins):
    assert pattri_ins.get_value('PSU.box0') == 1
    assert pattri_ins.get_value('PSU.csupos') == 444
    assert pattri_ins.get_value('PSU.box1', val='B') == 7126
    assert pattri_ins.get_value('PSU.spaces', insmode="Mode_B") == []
    assert pattri_ins.get_value('PSU.other', tag0='tag0_A', tag1='tag1_Y') == [-2]


def test_comp_psu_comps(pattri_ins):
    assert pattri_ins.get_property('PSU.BarL_1.csupos') == 444
    assert pattri_ins.get_property('PSU.BarL_2.csupos') == 444
    assert pattri_ins.get_property('PSU.BarR_1.csupos') == 444
    assert pattri_ins.get_property('PSU.BarR_2.csupos') == 444


def test_comp_robo_comps(pattri_ins):
    assert pattri_ins.get_property('robot.arm_1.angle') == 0
    assert pattri_ins.get_property('robot.arm_1.angle') == 0
    assert pattri_ins.get_property('robot.arm_9.active') == True
    pattri_ins.get_device('robot.arm_8').active = False
    assert pattri_ins.get_property('robot.arm_8.active') == False
    dev = pattri_ins.get_device('robot.arm_8')
    dev.get_property('active')