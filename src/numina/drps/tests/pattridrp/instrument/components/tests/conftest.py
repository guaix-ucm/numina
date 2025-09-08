
import astropy.io.fits as fits
import pytest

from numina.instrument.assembly import assembly_instrument
from numina.instrument.collection import load_paths_store


def load_from_tests(name, date_str):
    # pkg_paths = ["pattridrp.instrument.configs"]
    pkg_paths = ["numina.drps.tests.configs"]
    comp_store = load_paths_store(pkg_paths)
    ins = assembly_instrument(comp_store, name, date_str)
    return ins


@pytest.fixture
def pattri_ins():
    ins = load_from_tests("PATTRI", "2016-06-01T12:00:00.0")
    return ins


@pytest.fixture
def pattri_header():
    hdr = fits.Header()
    hdr["FILTER"] = "R"  # filter -> PATTRI.wheel.filter -> FILTER
    hdr["tag0"] = "tag0_A"
    hdr["tag1"] = "tag1_X"
    hdr["insmode"] = "Mode_B"
    hdr["val"] = "A"
    hdr["R00_ANGL"] = 0.0
    hdr["R01_ANGL"] = -0.3
    hdr["R02_ANGL"] = 0.1
    hdr["R03_ANGL"] = 0.4
    hdr["R04_ANGL"] = 0.4
    hdr["R05_ANGL"] = 0.1
    hdr["R06_ANGL"] = -0.3
    hdr["R00_ACTV"] = True
    hdr["R01_ACTV"] = True
    hdr["R02_ACTV"] = True
    hdr["R03_ACTV"] = False
    hdr["R04_ACTV"] = True
    hdr["R05_ACTV"] = True
    hdr["R06_ACTV"] = True
    return hdr


@pytest.fixture
def pattri_header2():
    hdr = fits.Header()
    hdr["POS"] = 3
    hdr["FILTER"] = "R"
    hdr["tag0"] = "tag0_A"
    hdr["tag1"] = "tag1_X"
    hdr["insmode"] = "Mode_A"
    hdr["R00_ANGL"] = 0.0
    hdr["R01_ANGL"] = -0.3
    hdr["R02_ANGL"] = 0.1
    hdr["R03_ANGL"] = 0.4
    hdr["R04_ANGL"] = 0.4
    hdr["R05_ANGL"] = 0.1
    hdr["R06_ANGL"] = -0.3
    hdr["R00_ACTV"] = False
    hdr["R01_ACTV"] = False
    hdr["R02_ACTV"] = False
    hdr["R03_ACTV"] = False
    hdr["R04_ACTV"] = False
    hdr["R05_ACTV"] = False
    hdr["R06_ACTV"] = False
    return hdr


@pytest.fixture
def pattri_header_err():
    hdr = fits.Header()
    hdr["POS"] = 3
    hdr["tag1"] = 1
    hdr["tag0"] = 1
    hdr["insmode"] = "Mode_A"
    hdr["filter"] = "U"
    return hdr


@pytest.fixture
def pattri_state():
    state = dict()
    state["PATTRI"] = {}
    state["PATTRI.wheel"] = {"label": "U"}
    state["PATTRI.robot.arm_2"] = {"angle": 1.2, "active": False}
    state["PATTRI.PSU"] = {"mode": "Mode_B"}
    state["PATTRI.PSU.BarL_1"] = {"pos": -8}
    return state
