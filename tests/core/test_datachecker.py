"""numina.core.config.check uses the checker of the DRP of the instrument"""

import pytest

import numina.core.dataload
import numina.core.pipelineload as pload
from tests.support.drpdef import make_drp

DRP_TEST1 = make_drp()


def check_test1(obj, astype=None, level=None):
    return ("TEST1", obj, astype)


@pytest.fixture
def drp_with_checker(drpmocker):
    def load_drp():
        drp = pload.drp_load_data("numina", DRP_TEST1)
        drp.checker = check_test1
        return drp

    drpmocker.add_drp("TEST1", load_drp)


def test_checker_of_drp(drp_with_checker):
    checker = numina.core.dataload.DataChecker()
    assert "TEST1" in checker
    assert checker("TEST1", "obj", astype="TYPE") == ("TEST1", "obj", "TYPE")


def test_no_checker(drpmocker):
    drpmocker.add_drp("TEST1", lambda: pload.drp_load_data("numina", DRP_TEST1))
    checker = numina.core.dataload.DataChecker()
    assert "TEST1" not in checker
    assert "OTHER" not in checker
    with pytest.warns(UserWarning, match="no function for OTHER"):
        assert checker("OTHER", "obj") is None


def test_register_deprecated(drpmocker):
    """A function registered is used if the DRP has no checker"""
    drpmocker.add_drp("TEST1", lambda: pload.drp_load_data("numina", DRP_TEST1))
    checker = numina.core.dataload.DataChecker()
    with pytest.warns(DeprecationWarning, match="DataChecker.register is deprecated"):

        @checker.register("TEST1")
        def check(obj, astype=None, level=None):
            return "registered"

    assert "TEST1" in checker
    assert checker("TEST1", "obj") == "registered"


def test_checker_of_drp_first(drp_with_checker):
    checker = numina.core.dataload.DataChecker()
    with pytest.warns(DeprecationWarning):
        checker.register("TEST1")(lambda obj, astype=None, level=None: "registered")
    assert checker("TEST1", "obj") == ("TEST1", "obj", None)
