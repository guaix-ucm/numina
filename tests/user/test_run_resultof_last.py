"""ResultOf with node='last' and a registry of reductions"""

import pytest

from numina.dal.registry import Registry

from tests.support.drpdef import make_drp
from .testdrp import DB_OPTIONS, SEEN, create_basedir, make_config, run_obs

DRP_TEST1 = make_drp({"bias": "tests.user.testdrp.BiasRecipe", "uselast": "tests.user.testdrp.UseLastRecipe"})


def bias(obid, image="image1.fits", **requirements):
    ob = {"id": obid, "mode": "bias", "instrument": "TEST1", "images": [image]}
    if requirements:
        ob["requirements"] = requirements
    return ob


USE_LAST = {"id": 100, "mode": "uselast", "instrument": "TEST1", "images": ["image1.fits"]}


@pytest.fixture
def basedir(drpmocker, tmp_path, monkeypatch):
    return create_basedir(drpmocker, tmp_path, monkeypatch, drp=DRP_TEST1)


def run(basedir, obs, *options, db=True):
    run_obs(make_config(basedir), obs, *(DB_OPTIONS if db else ()), *options)
    return SEEN


def bias_uuids(basedir):
    registry = Registry(str(basedir / "numina-db.json"))
    return [prod["uuid"] for prod in registry.products(type="MasterBias")]


def test_last_result(basedir):
    """The most recent result of the mode, of any observing block"""
    run(basedir, [bias(1), bias(2, image="image2.fits")])
    seen = run(basedir, [USE_LAST])
    assert seen == bias_uuids(basedir)[-1:]


def test_last_result_not_bad(basedir):
    run(basedir, [bias(1), bias(2, image="image2.fits", quality="BAD")])
    seen = run(basedir, [USE_LAST])
    first, second = bias_uuids(basedir)
    assert seen == [first]


def test_last_result_same_profile(basedir):
    """A more recent result of another instrument profile is not used"""
    run(basedir, [bias(1), bias(2, image="image3.fits")])
    seen = run(basedir, [USE_LAST])
    first, other_profile = bias_uuids(basedir)
    assert seen == [first]


def test_no_result(basedir):
    with pytest.raises(ValueError, match="Required 'master_bias'"):
        run(basedir, [USE_LAST])


def test_without_registry(basedir):
    run(basedir, [bias(1)], db=False)
    with pytest.raises(ValueError, match="Required 'master_bias'"):
        run(basedir, [USE_LAST], db=False)
