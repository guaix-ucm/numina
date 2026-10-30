#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import warnings
import astropy.io.fits as fits
import pytest

warnings.warn(
    "numina.testing.plugins_pattri is deprecated and will be removed in the next version of numina, "
    "it is only used by the tests of numina",
    DeprecationWarning,
    stacklevel=2,
)


@pytest.fixture
def pattri_header():
    """Return a FITS header of the test instrument PATTRI."""
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
    """Return a FITS header of the test instrument PATTRI, with all the robot arms inactive."""
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
    """Return a FITS header of the test instrument PATTRI with integer values in TAG0 and TAG1."""
    hdr = fits.Header()
    hdr["POS"] = 3
    hdr["tag1"] = 1
    hdr["tag0"] = 1
    hdr["insmode"] = "Mode_A"
    hdr["filter"] = "U"
    return hdr


@pytest.fixture
def pattri_state():
    """Return a dictionary with the state of the components of the test instrument PATTRI."""
    state = dict()
    state["PATTRI"] = {}
    state["PATTRI.wheel"] = {"label": "U"}
    state["PATTRI.robot.arm_2"] = {"angle": 1.2, "active": False}
    state["PATTRI.PSU"] = {"mode": "Mode_B"}
    state["PATTRI.PSU.BarL_1"] = {"pos": -8}
    return state
