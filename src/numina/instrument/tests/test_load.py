import pytest

from ..assembly import assembly_instrument
from ..collection import load_paths_store


def load_from_tests(name, date_str):
    # pkg_paths = ["pattridrp.instrument.configs"]
    pkg_paths = ["numina.drps.tests.configs"]
    comp_store = load_paths_store(pkg_paths)
    ins = assembly_instrument(comp_store, name, date_str)
    return ins


def test_ins_load_fail():

    with pytest.raises(ValueError):
        load_from_tests("PATTRI", "2006-06-01T12:00:00.0")

    with pytest.raises(ValueError):
        load_from_tests("WRONG_NAME", "2016-06-01T12:00:00.0")
