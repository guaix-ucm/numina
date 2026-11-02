"""Index of the installed calibration files, from their envelopes"""

import datetime
import json
import logging
import pathlib

import astropy.io.fits as fits
import numpy
import pytest

import numina.instrument.assembly as asb
from numina.keydef import QueryAttribute
from numina.dal.calibindex import index_calibrations
from numina.dal.dictdal import HybridDAL
from numina.dal.registry import MemoryStore, Registry
from numina.exceptions import NoResultFound
from numina.types.linescatalog import LinesCatalog
from tests.support.drptest import create_drp_test
from tests.support.recipes import MasterBias

PROFILE = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"
OTHER_PROFILE = "9f3fecbf-376c-47ae-a188-313b7d829104"

OBLOCKS = [dict(id=2, instrument="TEST1", mode="image", images=[], children=[])]

LINES = """\
# %ECSV 1.0
# ---
# datatype:
# - {{name: wavelength, datatype: float64}}
# - {{name: flag, datatype: int64}}
# - {{name: intensity, datatype: int64}}
# meta: !!omap
# - {{envelope: 1}}
# - {{type: LinesCatalog}}
# - {{format: {format}}}
# - {{instrument: TEST1}}
# - {{uuid: {uuid}}}
# - tags: {{vph: {vph}}}
# schema: astropy-2.0
wavelength flag intensity
{wave} 1 100
"""


class VPHLinesCatalog(LinesCatalog):
    __tags__ = {"vph": QueryAttribute("vph", str)}

    def name(self):
        return "LinesCatalog"


def write_lines(directory, name, vph, wave, uuid, format=2):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(LINES.format(vph=vph, wave=wave, uuid=uuid, format=format))
    return path


def write_bias(path, profile=PROFILE, envelope=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(numpy.zeros((2, 2)))
    hdu.header["INSTRUME"] = "TEST1"
    hdu.header["UUID"] = path.stem
    hdu.header["INSCONF"] = profile
    if envelope:
        hdu.header["NUMENV"] = 1
        hdu.header["NUMTYPE"] = "MasterBias"
        hdu.header["NUMFMT"] = 1
    hdu.writeto(path)
    return path


@pytest.fixture
def calibs(tmp_path):
    """Calibrations of TEST1, in any directory"""
    base = tmp_path / "calib" / "TEST1"
    write_lines(base / "catalogs", "lr.ecsv", "LR-U", 3600.0, "lines-lr")
    write_lines(base / "catalogs", "hr.ecsv", "HR-R", 6500.0, "lines-hr")
    write_bias(base / "anywhere" / "bias-1.fits")
    write_bias(base / "anywhere" / "bias-other.fits", profile=OTHER_PROFILE)
    (base / "README.md").write_text("no envelope\n")
    (base / ".hidden").mkdir()
    write_bias(base / ".hidden" / "bias-hidden.fits")
    return base


@pytest.fixture
def drps():
    return create_drp_test(["drptest1.yaml"])


@pytest.fixture
def components():
    return asb.load_paths_store(["tests.support.drps.configs"])


def make_dal(drps, components, tmp_path):
    return HybridDAL(drps, str(tmp_path / "calib"), OBLOCKS, {}, basedir=str(tmp_path), components=components)


def test_index(calibs):
    index = index_calibrations(str(calibs))
    products = {prod["uuid"]: prod for prod in index.products()}
    assert sorted(products) == ["bias-1", "bias-other", "lines-hr", "lines-lr"]
    lines = products["lines-lr"]
    assert lines["origin"] == "installed"
    assert lines["profile"] is None
    assert lines["format"] == 2
    assert lines["tags"] == {"vph": "LR-U"}
    assert lines["content"] == str(calibs / "catalogs" / "lr.ecsv")
    assert products["bias-1"]["profile"] == PROFILE


def test_index_invalid_envelope(calibs, caplog):
    path = calibs / "bad.fits"
    write_bias(path)
    with fits.open(path, mode="update") as hdul:
        del hdul[0].header["NUMTYPE"]
    with caplog.at_level(logging.WARNING, logger="numina.dal.calibindex"):
        index = index_calibrations(str(calibs))
    assert len(index.products()) == 4
    assert "bad.fits not indexed" in caplog.text


def test_search_by_tags(drps, components, tmp_path, calibs):
    dal = make_dal(drps, components, tmp_path)
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)
    obsres.tags = {"vph": "HR-R"}

    result = dal.search_product("lines_catalog", VPHLinesCatalog(), obsres)

    numpy.testing.assert_array_equal(result.content, [[6500.0, 1, 100]])
    assert result.id == "lines-hr"
    assert dal.sources["lines_catalog"] == {
        "source": "calibrations index",
        "path": str(calibs / "catalogs" / "hr.ecsv"),
        "uuid": "lines-hr",
        "tags": {"vph": "HR-R"},
    }


def test_search_by_profile(drps, components, tmp_path, calibs):
    """The product with the profile of the observation, wherever it is"""
    dal = make_dal(drps, components, tmp_path)
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)

    result = dal.search_product("master_bias", MasterBias(), obsres)

    assert result.content.filename == str(calibs / "anywhere" / "bias-1.fits")


def test_index_before_directory(drps, components, tmp_path, calibs):
    """The files with envelope are preferred to those found by their path"""
    write_bias(calibs / PROFILE / "MasterBias" / "mbias.fits", envelope=False)
    dal = make_dal(drps, components, tmp_path)
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)

    result = dal.search_product("master_bias", MasterBias(), obsres)

    assert dal.sources["master_bias"]["source"] == "calibrations index"
    assert result.content.filename == str(calibs / "anywhere" / "bias-1.fits")


def test_directory_without_envelopes(drps, components, tmp_path):
    """Without files with envelope, the file is found by its path"""
    path = write_bias(tmp_path / "calib" / "TEST1" / PROFILE / "MasterBias" / "mbias.fits", envelope=False)
    dal = make_dal(drps, components, tmp_path)
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)

    result = dal.search_product("master_bias", MasterBias(), obsres)

    assert dal.sources["master_bias"]["source"] == "calibrations directory"
    assert result.content.filename == str(path)


def test_unreadable_format(drps, components, tmp_path, calibs):
    write_lines(calibs / "catalogs", "new.ecsv", "MR-B", 4500.0, "lines-new", format=3)
    dal = make_dal(drps, components, tmp_path)
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)
    obsres.tags = {"vph": "MR-B"}

    with pytest.raises(NoResultFound, match=r"this version reads formats 1, 2, found .*new.ecsv \(format 3\)"):
        dal.search_product("lines_catalog", VPHLinesCatalog(), obsres)


def test_index_built_once(drps, components, tmp_path, calibs):
    dal = make_dal(drps, components, tmp_path)
    assert dal.calibrations_index("TEST1") is dal.calibrations_index("TEST1")
    assert dal.calibrations_index("OTHER") is None


def registry_with(*docs):
    registry = Registry(None, store=MemoryStore())
    for doc in docs:
        base = {"instrument": "TEST1", "type": "MasterBias", "qc": "GOOD", "tags": {}, "priority": 0}
        registry.store.insert("products", dict(base, **doc))
    return registry


def test_select_without_profile():
    registry = registry_with({"uuid": "a", "profile": OTHER_PROFILE}, {"uuid": "b", "profile": None})
    assert registry.select_product("TEST1", PROFILE)["uuid"] == "b"
    # The most recent among those of the same priority
    assert registry.select_product("TEST1", OTHER_PROFILE)["uuid"] == "b"


@pytest.mark.parametrize(
    "date, expected",
    [
        (datetime.datetime(2016, 6, 1), "old"),
        (datetime.datetime(2017, 1, 1), "new"),
        (datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC), "new"),
        (None, "new"),
    ],
)
def test_select_validity(date, expected):
    registry = registry_with(
        {"uuid": "old", "profile": None, "validity": {"start": "2015-01-01", "end": "2017-01-01"}},
        {"uuid": "new", "profile": None, "validity": {"start": "2017-01-01"}},
    )
    assert registry.select_product("TEST1", PROFILE, date=date)["uuid"] == expected


def test_select_outside_validity():
    registry = registry_with({"uuid": "old", "profile": None, "validity": {"end": "2017-01-01"}})
    assert registry.select_product("TEST1", PROFILE, date=datetime.datetime(2018, 1, 1)) is None


def test_cache(calibs, calibindex_cache, monkeypatch):
    """The second index reads only the new and changed files"""
    from numina.dal import calibindex

    index_calibrations(str(calibs))
    assert len(list(calibindex_cache.glob("*.json"))) == 1

    read = []
    original = calibindex.read_envelope

    def counting(path):
        read.append(path)
        return original(path)

    monkeypatch.setattr(calibindex, "read_envelope", counting)
    index = index_calibrations(str(calibs))
    assert read == []
    assert len(index.products()) == 4

    write_lines(calibs / "catalogs", "mr.ecsv", "MR-B", 4500.0, "lines-mr")
    lines_lr = calibs / "catalogs" / "lr.ecsv"
    lines_lr.write_text(lines_lr.read_text().replace("3600.0", "3601.00"))
    (calibs / "catalogs" / "hr.ecsv").unlink()
    index = index_calibrations(str(calibs))
    assert sorted(read) == [str(calibs / "catalogs" / "lr.ecsv"), str(calibs / "catalogs" / "mr.ecsv")]
    assert sorted(prod["uuid"] for prod in index.products()) == ["bias-1", "bias-other", "lines-lr", "lines-mr"]


def test_cache_other_version(calibs, calibindex_cache, monkeypatch):
    """Each version of numina has its own cache, they do not discard each other"""
    import numina
    from numina.dal import calibindex

    version = numina.__version__
    index_calibrations(str(calibs))
    read = []
    original = calibindex.read_envelope

    def counting(path):
        read.append(path)
        return original(path)

    monkeypatch.setattr(calibindex, "read_envelope", counting)
    monkeypatch.setattr(numina, "__version__", "0.0.1")
    index_calibrations(str(calibs))
    # All the files are read: README.md, the 2 catalogs and the 2 bias
    assert len(read) == 5
    assert len(list(calibindex_cache.glob("*.json"))) == 2

    read.clear()
    monkeypatch.setattr(numina, "__version__", version)
    index = index_calibrations(str(calibs))
    assert read == []
    assert len(index.products()) == 4


def test_cache_unused_pruned(calibs, tmp_path, calibindex_cache):
    """The caches not used in CACHE_MAX_AGE days are removed, the used ones are kept"""
    import os
    import time

    from numina.dal import calibindex

    other = tmp_path / "other"
    write_bias(other / "bias-other.fits")
    index_calibrations(str(calibs))
    index_calibrations(str(other))
    path_calibs = calibindex.cache_path(str(calibs), str(calibindex_cache))
    path_other = calibindex.cache_path(str(other), str(calibindex_cache))
    old = time.time() - (calibindex.CACHE_MAX_AGE + 1) * 86400
    for path in (path_calibs, path_other):
        os.utime(path, (old, old))
    # calibs is used, its cache is touched
    index_calibrations(str(calibs))
    assert os.path.getmtime(path_calibs) > old
    # A cache is written, the cache of other is removed
    write_lines(calibs / "catalogs", "mr.ecsv", "MR-B", 4500.0, "lines-mr")
    index_calibrations(str(calibs))
    assert sorted(calibindex_cache.glob("*.json")) == [pathlib.Path(path_calibs)]


def test_cache_invalid_envelope_warns(calibs, caplog):
    """The error is kept in the cache, and the warning is given again"""
    path = calibs / "bad.fits"
    write_bias(path)
    with fits.open(path, mode="update") as hdul:
        del hdul[0].header["NUMTYPE"]
    index_calibrations(str(calibs))
    with caplog.at_level(logging.WARNING, logger="numina.dal.calibindex"):
        index_calibrations(str(calibs))
    assert "bad.fits not indexed" in caplog.text


def test_cache_pruned(tmp_path, calibindex_cache):
    old = tmp_path / "old"
    write_bias(old / "bias-old.fits")
    index_calibrations(str(old))
    write_bias(tmp_path / "new" / "bias-new.fits")
    for path in old.iterdir():
        path.unlink()
    old.rmdir()
    index_calibrations(str(tmp_path / "new"))
    caches = list(calibindex_cache.glob("*.json"))
    assert [json.loads(path.read_text())["directory"] for path in caches] == [str(tmp_path / "new")]


def test_cache_not_writable(calibs, tmp_path, caplog):
    cache_dir = tmp_path / "readonly"
    cache_dir.mkdir()
    cache_dir.chmod(0o500)
    try:
        index = index_calibrations(str(calibs), cache_dir=str(cache_dir))
    finally:
        cache_dir.chmod(0o700)
    assert len(index.products()) == 4
    assert list(cache_dir.iterdir()) == []


def test_without_cache(calibs, calibindex_cache):
    index = index_calibrations(str(calibs), cache=False)
    assert len(index.products()) == 4
    assert list(calibindex_cache.iterdir()) == []
