"""The automatic selection of products: priority, then the closest observation date, then the most recent"""

import datetime

import astropy.io.fits as fits
import pytest

from numina.core.oresult import ObservationResult
from numina.dal.dictdal import BaseHybridDAL
from numina.dal.registry import Registry
from numina.datamodel import DataModel
from numina.types.dataframe import DataFrame


@pytest.fixture
def registry(tmp_path):
    return Registry(str(tmp_path / "numina-db.json"), basedir=str(tmp_path))


def add_product(registry, time_obs, priority=None, qc="GOOD"):
    doc = {
        "type": "MasterBias",
        "instrument": "TEST1",
        "profile": "P",
        "qc": qc,
        "time_obs": time_obs,
        "content": "master_bias.fits",
        "uuid": None,
    }
    if priority is not None:
        doc["priority"] = priority
    with registry.store.transaction():
        return registry.store.insert("products", doc)


def select(registry, date=None):
    prod = registry.select_product("TEST1", "P", date=date)
    return None if prod is None else prod["id"]


def test_most_recent_without_date(registry):
    add_product(registry, "2017-01-01T00:00:00")
    second = add_product(registry, "2010-01-01T00:00:00")
    assert select(registry) == second


def test_closest_date(registry):
    jan = add_product(registry, "2017-01-01T00:00:00")
    add_product(registry, "2017-06-01T00:00:00")
    add_product(registry, "2010-01-01T00:00:00")
    assert select(registry, datetime.datetime(2017, 1, 5)) == jan


def test_same_distance_most_recent(registry):
    add_product(registry, "2017-01-01T00:00:00")
    second = add_product(registry, "2017-01-03T00:00:00")
    assert select(registry, datetime.datetime(2017, 1, 2)) == second


def test_unknown_date_last(registry):
    dated = add_product(registry, "2010-01-01T00:00:00")
    add_product(registry, None)
    assert select(registry, datetime.datetime(2017, 1, 1)) == dated


def test_priority_first(registry):
    old = add_product(registry, "2010-01-01T00:00:00", priority=1)
    # the products of older registries have no priority, it is 0
    add_product(registry, "2017-01-01T00:00:00")
    assert select(registry, datetime.datetime(2017, 1, 1)) == old
    newer = add_product(registry, "2017-01-01T00:00:00", priority=2)
    assert select(registry, datetime.datetime(2017, 1, 1)) == newer


def test_negative_priority(registry):
    add_product(registry, "2017-01-01T00:00:00", priority=-1)
    other = add_product(registry, "2010-01-01T00:00:00")
    assert select(registry, datetime.datetime(2017, 1, 1)) == other


def test_bad_qc_excluded(registry):
    add_product(registry, "2017-01-01T00:00:00", priority=5, qc="BAD")
    good = add_product(registry, "2010-01-01T00:00:00")
    assert select(registry) == good


def test_timezones(registry):
    utc = add_product(registry, "2017-01-01T00:00:00+00:00")
    add_product(registry, "2016-01-01T00:00:00")
    assert select(registry, datetime.datetime(2017, 1, 1)) == utc


def test_set_priority(registry):
    first = add_product(registry, "2017-01-01T00:00:00")
    add_product(registry, "2017-01-01T00:00:00")
    assert registry.set_priority(first, 3) == first
    assert registry.store.get("products", first)["priority"] == 3
    assert select(registry) == first
    with pytest.raises(ValueError, match="no product with id 99"):
        registry.set_priority(99, 1)


class FakeDRP:
    datamodel = DataModel()


@pytest.mark.parametrize(
    "date_obs, expected",
    [("2017-01-05T10:00:00", datetime.datetime(2017, 1, 5, 10)), (None, None)],
)
def test_observation_date_of_oblock(date_obs, expected):
    hdu = fits.PrimaryHDU()
    if date_obs is not None:
        hdu.header["DATE-OBS"] = date_obs
    obsres = ObservationResult()
    obsres.frames = [DataFrame(frame=fits.HDUList([hdu]))]
    assert BaseHybridDAL._observation_date(obsres, FakeDRP()) == expected


def test_observation_date_without_frames():
    assert BaseHybridDAL._observation_date(ObservationResult(), FakeDRP()) is None
