from datetime import datetime

import pytest

from ...assembly import (
    find_element,
    ElementEnum,
    find_instrument,
    process_setup,
    process_properties,
    process_components,
)
from ...collection import load_paths_store
from ...generic import ComponentGeneric


def test_ins_prefix(pattri_ins):

    devpath1 = pattri_ins.get_device("PATTRI.PSU")
    devpath2 = pattri_ins.get_device("PSU")

    assert isinstance(devpath1, ComponentGeneric)

    assert devpath1 is devpath2


def test_all_comps(pattri_ins):

    id_names = ["PSU", "detector", "wheel"]
    comp_names = ["PSU", "Detector_Model1", "Wheel_Model1"]
    comp_uuids = [
        "41fa8884-2ea3-42c3-98a6-a17a29c990cf",
        "510efa03-c266-4202-8a4d-d5799af49d9d",
        "27614307-783f-4c1c-bd9c-165ff9cec1c5",
    ]
    for id_n, c_name, c_uuid in zip(id_names, comp_names, comp_uuids):
        dev = pattri_ins.get_device(id_n)
        assert dev.origin.name == c_name
        assert str(dev.origin.uuid) == c_uuid


@pytest.fixture
def comp_store():
    pkg_paths = ["numina.drps.tests.configs"]
    # pkg_paths = ["pattridrp.instrument.configs"]
    comp_store = load_paths_store(pkg_paths)
    return comp_store


@pytest.mark.parametrize(
    "date_val", ["2015-05-12T03:04:21", datetime(2015, 5, 12, 3, 4, 21)]
)
def test_find_element1(comp_store, date_val):
    etype = ElementEnum.ELEM_COMPONENT
    el = find_element(
        comp_store,
        etype,
        "27614307-783f-4c1c-bd9c-165ff9cec1c5",
        date_val,
        by_key="uuid",
    )
    assert isinstance(el, dict)


@pytest.mark.parametrize(
    "date_val", ["2015-05-12T03:04:21", datetime(2015, 5, 12, 3, 4, 21)]
)
def test_find_instrument(comp_store, date_val):
    uuid_str = "43273e8c-4071-4a73-a6b4-40c2f07cf054"
    el = find_instrument(comp_store, uuid_str, date_val, by_key="uuid")
    assert isinstance(el, dict)


def test_process_setup(comp_store):
    date_str = "2018-12-12T03:04:21"
    uuid_str = "a7358f24-6ce7-4851-a197-d6515e0592f5"
    block = [{"uuid": uuid_str, "id": "test_block"}]
    aa = process_setup(comp_store, setup_block=block, setup_id="test", date=date_str)
    assert aa.name == "test"
    assert aa.origin is None
    assert "a" in aa.values
    assert "b" in aa.values
    assert isinstance(aa.values["a"], dict)
    assert isinstance(aa.values["b"], dict)


def test_process_setup_error1(comp_store):
    date_str = "2018-12-12T03:04:21"
    block = [{"xxx": "invalid"}]
    with pytest.raises(ValueError):
        process_setup(comp_store, setup_block=block, setup_id="test", date=date_str)


def test_process_properties_error1(comp_store):
    date_str = "2018-12-12T03:04:21"
    block = [{"xxx": "invalid", "id": "test_block"}]
    with pytest.raises(ValueError):
        process_properties(comp_store, prop_block=block, date=date_str)


def test_process_components_error1(comp_store):
    date_str = "2018-12-12T03:04:21"
    block = [{"xxx": "invalid"}]
    with pytest.raises(ValueError):
        process_components(comp_store, comp_block=block, date=date_str)


def test_process_properties_error2(comp_store):
    date_str = "2018-12-12T03:04:21"
    block = [{"one_of": [], "id": "test_block"}]
    with pytest.raises(ValueError):
        process_properties(comp_store, prop_block=block, date=date_str)
