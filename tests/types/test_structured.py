"""Structured calibrations of numina.types.structured"""

import datetime
import json

import pytest

import numina.types.structured as structured
from numina.types.qc import QC


class StructuredCal(structured.BaseStructuredCalibration):
    pass


class Mode:
    key = "test_mode"


class Recipe:
    __version__ = "1.2"
    instrument = "TEST"
    mode = Mode()


@pytest.fixture
def calib():
    obj = StructuredCal(instrument="TEST")
    obj.tags = {"filter": "J"}
    obj.quality_control = QC.GOOD
    return obj


ORIGIN = {
    "block_uuid": "b1",
    "insconf_uuid": "i1",
    "observation_date": "2026-01-02T03:04:05",
    "frames": ["r1.fits", "r2.fits"],
}


def test_structured(calib):
    assert calib.calibid == f"uuid:{calib.uuid}"
    assert calib.default is None
    assert calib.meta_info["instrument_name"] == "TEST"
    assert str(calib) == f"StructuredCal(instrument=TEST, uuid={calib.uuid})"
    assert str(StructuredCal()) == "StructuredCal()"


def test_extract_tags(calib):
    assert StructuredCal().extract_tags(calib) == {"filter": "J"}


def test_getstate(calib):
    state = calib.__getstate__()
    assert state["instrument"] == "TEST"
    assert state["tags"] == {"filter": "J"}
    assert state["uuid"] == calib.uuid
    assert state["type"] == "StructuredCal"
    assert state["type_fqn"] == "tests.types.test_structured.StructuredCal"
    assert state["quality_control"] == QC.GOOD


def check_copy(obj, calib):
    assert type(obj) is StructuredCal
    assert obj.uuid == calib.uuid
    assert obj.instrument == "TEST"
    assert obj.tags == {"filter": "J"}
    assert obj.quality_control == QC.GOOD
    assert obj.meta_info == calib.meta_info


def test_writeto_open(calib, tmp_path):
    filename = tmp_path / "calib.json"
    calib.writeto(filename)

    check_copy(structured.open(filename), calib)
    with open(filename) as fd:
        check_copy(structured.load(fd), calib)


def test_loads():
    data = {"a": 1}
    assert structured.loads(data) is data


def test_datatype_dump_load(calib, tmp_path):
    filename = StructuredCal._datatype_dump(calib, str(tmp_path / "calib"))
    assert filename == str(tmp_path / "calib.json")
    check_copy(StructuredCal._datatype_load(filename), calib)

    with pytest.raises(OSError):
        StructuredCal._datatype_load(str(tmp_path / "missing.json"))


def test_update_metadata(calib):
    calib.update_metadata(Recipe())
    assert calib.meta_info["mode_name"] == "test_mode"
    assert calib.meta_info["instrument_name"] == "TEST"
    assert calib.meta_info["recipe_name"] == "Recipe"
    assert calib.meta_info["recipe_version"] == "1.2"

    recipe = Recipe()
    recipe.mode = "other_mode"
    calib.update_metadata(recipe)
    assert calib.meta_info["mode_name"] == "other_mode"


def test_meta(calib):
    meta = calib.meta
    assert meta["instrument"] == "TEST"
    assert meta["uuid"] == calib.uuid
    assert meta["tags"] == {"filter": "J"}
    assert meta["type"] == "StructuredCal"
    assert meta["mode"] == "unknown"
    assert meta["quality_control"] == "GOOD"
    assert meta["observation_date"] == datetime.datetime(1970, 1, 1)
    assert meta["origin"] == {}


def test_meta_origin(calib):
    calib.update_metadata(Recipe())
    calib.update_metadata_origin(ORIGIN)

    origin = calib.meta_info["origin"]
    assert origin["block_uuid"] == "b1"
    assert origin["insconf_uuid"] == "i1"
    assert origin["date_obs"] == "2026-01-02T03:04:05"
    assert origin["frames"] == ["r1.fits", "r2.fits"]

    meta = calib.meta
    assert meta["mode"] == "test_mode"
    assert meta["observation_date"] == datetime.datetime(2026, 1, 2, 3, 4, 5)
    assert meta["origin"] is origin


def test_extract_db_info_object(calib):
    assert StructuredCal().extract_db_info(calib, []) == calib.meta


@pytest.mark.parametrize("with_origin", [True, False])
def test_extract_db_info_file(calib, tmp_path, with_origin):
    if with_origin:
        calib.update_metadata(Recipe())
        calib.update_metadata_origin(ORIGIN)
    filename = tmp_path / "calib.json"
    calib.writeto(filename)

    info = StructuredCal().extract_db_info(str(filename), [])

    assert info["instrument"] == "TEST"
    assert info["uuid"] == calib.uuid
    assert info["tags"] == {"filter": "J"}
    assert info["type"] == "StructuredCal"
    assert info["quality_control"] == "GOOD"
    if with_origin:
        assert info["mode"] == "test_mode"
        assert info["observation_date"] == datetime.datetime(2026, 1, 2, 3, 4, 5)
        assert info["origin"]["block_uuid"] == "b1"
    else:
        assert info["mode"] == "unknown"
        assert info["observation_date"] == datetime.datetime(1970, 1, 1)
        assert info["origin"] == {}


def test_extract_db_info_no_meta_info(tmp_path):
    filename = tmp_path / "calib.json"
    state = {"instrument": "TEST", "uuid": "u1", "tags": {}, "type": "StructuredCal"}
    filename.write_text(json.dumps(state))

    info = StructuredCal().extract_db_info(str(filename), [])

    assert info["uuid"] == "u1"
    assert info["quality_control"] == QC.UNKNOWN
    assert info["observation_date"] == datetime.datetime(1970, 1, 1)


def test_extract_db_info_missing(tmp_path):
    with pytest.raises(OSError):
        StructuredCal().extract_db_info(str(tmp_path / "missing.json"), [])
