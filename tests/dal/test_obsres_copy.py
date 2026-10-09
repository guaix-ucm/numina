"""The observation result does not modify the stored observing block"""

import astropy.io.fits as fits
import pytest

import numina.instrument.assembly as asb
from numina.testing.drptest import create_drp_test

from numina.dal.dictdal import HybridDAL


@pytest.fixture
def dal():
    drps = create_drp_test(["drptest1.yaml"])
    store = asb.load_paths_store(["numina.testing.drps.configs"])
    base = HybridDAL(drps, "", [], {"products": {}, "requirements": {}}, components=store)
    hdr = fits.Header()
    hdr["INSTRUME"] = "TEST1"
    hdr["DATE-OBS"] = "2017-10-01T00:00:00"
    obs = dict(
        id=1,
        instrument="TEST1",
        mode="image",
        images=[fits.HDUList([fits.PrimaryHDU(header=hdr)])],
        requirements={"a": 1},
        results={"r1": 1},
        labels={"l1": 1},
    )
    base.add_obs([obs])
    return base


def test_ob_table_not_modified(dal):
    obsres = dal.obsres_from_oblock_id(1)
    # as run_reduce does
    obsres.requirements.update({"b": 2})
    obsres.results["r2"] = 2
    obsres.labels["l2"] = 2

    ob = dal.ob_table[1]
    assert ob["requirements"] == {"a": 1}
    assert ob["results"] == {"r1": 1}
    assert ob["labels"] == {"l1": 1}


def test_oblock_not_modified(dal):
    oblock = dal.oblock_from_id(1)
    obsres = dal.obsres_from_oblock(oblock)
    obsres.requirements.update({"b": 2})

    assert obsres.__dict__ is not oblock.__dict__
    assert obsres.profile == "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"
    assert oblock.profile == "00000000-0000-0000-0000-000000000000"
    assert oblock.configuration == "default"
    assert oblock.requirements == {"a": 1}
