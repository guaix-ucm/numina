"""Searches of products, parameters and results in HybridDAL"""

import io
import json
import logging

import astropy.io.fits as fits
import numpy
import pytest
import yaml

import numina.instrument.assembly as asb
from numina.core.query import ResultOf
from numina.dal.dictdal import BaseHybridDAL, Dict2DAL, HybridDAL
from numina.exceptions import NoResultFound
from tests.support.drptest import create_drp_test
from tests.support.recipes import MasterBias

PROFILE = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"

OBLOCKS = [
    dict(id=1, instrument="TEST1", mode="bias", images=[], children=[]),
    dict(id=2, instrument="TEST1", mode="image", images=[], children=[]),
    dict(id=3, instrument="TEST1", mode="image", images=[], children=[]),
    dict(id=10, instrument="TEST1", mode="raiz", images=[], children=[1, 2]),
]


@pytest.fixture
def drps():
    return create_drp_test(["drptest1.yaml"])


@pytest.fixture
def components():
    return asb.load_paths_store(["tests.support.drps.configs"])


@pytest.fixture
def calibdir(tmp_path):
    """Directory of calibrations with a master bias"""
    directory = tmp_path / "calib" / "TEST1" / PROFILE / "MasterBias"
    directory.mkdir(parents=True)
    fits.PrimaryHDU(numpy.zeros((2, 2))).writeto(directory / "mbias.fits")
    return directory


def make_dal(drps, components, tmp_path, base=None, extra_data=None):
    return HybridDAL(
        drps,
        str(tmp_path / "calib"),
        OBLOCKS,
        base if base is not None else {},
        extra_data=extra_data,
        basedir=str(tmp_path),
        components=components,
    )


def write_result(tmp_path, obsid, values):
    directory = tmp_path / f"obsid{obsid}_results"
    directory.mkdir()
    (directory / "result.json").write_text(json.dumps({"values": values}))


def test_product_in_table(drps, components, tmp_path, calibdir):
    base = {"products": {"TEST1": {PROFILE: [{"id": 7, "type": "MasterBias", "tags": {}, "content": "other.fits"}]}}}
    dal = make_dal(drps, components, tmp_path, base)
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)

    result = dal.search_product("master_bias", MasterBias(), obsres)

    assert result.id == 7
    assert result.content.filename == "other.fits"
    assert dal.sources["master_bias"] == {"source": "control file"}


def test_product_in_table_without_content(drps, components, tmp_path, calibdir):
    base = {"products": {"TEST1": {PROFILE: [{"id": 7, "type": "MasterBias", "tags": {}}]}}}
    dal = make_dal(drps, components, tmp_path, base)
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)

    result = dal.search_product("master_bias", MasterBias(), obsres)

    assert result.id == 7
    assert result.content.filename == str(calibdir / "mbias.fits")


def test_product_in_directory(drps, components, tmp_path, calibdir):
    dal = make_dal(drps, components, tmp_path)
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)

    result = dal.search_product("master_bias", MasterBias(), obsres)

    path = str(calibdir / "mbias.fits")
    assert result.id == 0
    assert result.content.filename == path
    assert dal.sources["master_bias"] == {"source": "calibrations directory", "path": (path, 0)}


def test_product_in_directory_gtc(drps, components, tmp_path, calibdir):
    path = str(calibdir / "mbias.fits")
    element = {"name": "master_bias", "typeid": 45, "value": {"path": path}}
    (calibdir / "result.json").write_text(json.dumps({"elements": [element]}))
    dal = make_dal(drps, components, tmp_path)
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)

    result = dal.search_product("master_bias", MasterBias(), obsres)

    assert result.content.filename == path
    assert dal.sources["master_bias"]["path"] == (str(calibdir / "result.json"), 1)


def test_product_not_found(drps, components, tmp_path):
    dal = make_dal(drps, components, tmp_path)
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)

    with pytest.raises(NoResultFound):
        dal.search_product("master_bias", MasterBias(), obsres)


def test_product_extra_data(drps, components, tmp_path):
    dal = make_dal(drps, components, tmp_path, extra_data={"master_bias": "given.fits"})
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)

    result = dal.search_product("master_bias", MasterBias(), obsres)

    assert result.content.filename == "given.fits"
    assert dal.sources["master_bias"] == {"source": "command line"}


def parameters():
    return {"TEST1": {PROFILE: {"default": {"image": [{"name": "gain", "tags": {}, "content": 3.5}]}}}}


@pytest.mark.parametrize("key", ["parameters", "requirements"])
def test_search_parameter(drps, components, tmp_path, key):
    dal = make_dal(drps, components, tmp_path, base={key: parameters()})
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)

    assert dal.search_parameter("gain", float, obsres).content == 3.5
    assert dal.sources["gain"] == {"source": "control file"}

    with pytest.raises(NoResultFound, match="name other compatible with tags"):
        dal.search_parameter("other", float, obsres)


def test_search_parameter_extra_data(drps, components, tmp_path):
    dal = make_dal(drps, components, tmp_path, base={"parameters": parameters()}, extra_data={"gain": 1.5})
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)

    assert dal.search_parameter("gain", float, obsres).content == 1.5
    assert dal.sources["gain"] == {"source": "command line"}


def test_obsres_reserved_mode(drps, components, tmp_path):
    dal = make_dal(drps, components, tmp_path)
    obsres = dal.obsres_from_oblock_id(10, profile=PROFILE)
    assert obsres.mode == "raiz"
    assert obsres.children == [1, 2]


def test_obsres_without_drp(drps, components, tmp_path):
    dal = make_dal(drps, components, tmp_path)
    dal.add_obs([dict(id=20, instrument="OTHER", mode="bias", images=[], children=[])])
    with pytest.raises(ValueError, match="no DRP for instrument OTHER"):
        dal.obsres_from_oblock_id(20)


def test_add_obs_repeated(drps, components, tmp_path, caplog):
    dal = make_dal(drps, components, tmp_path)
    with caplog.at_level(logging.WARNING, logger="numina.dal.dictdal"):
        dal.add_obs([dict(OBLOCKS[0], mode="image")])
    assert "oblock_id=1 is already in table" in caplog.text
    assert dal.ob_table[1]["mode"] == "bias"


def test_assembly_instrument(drps, components, tmp_path):
    dal = make_dal(drps, components, tmp_path)
    instrument = dal.assembly_instrument(PROFILE, None, by_key="uuid")
    assert instrument.name == "TEST1"


def test_dump(drps, components, tmp_path):
    dal = make_dal(drps, components, tmp_path, base={"parameters": parameters()})
    state = dal.dump_data()
    assert state["version"] == 1
    assert state["rootdir"] == str(tmp_path / "calib")
    assert state["ob_ids"] == [1, 2, 3, 10]
    assert state["requirements"] == parameters()

    fp = io.StringIO()
    dal.dump(fp)
    assert yaml.safe_load(fp.getvalue())["ob_ids"] == [1, 2, 3, 10]


def test_new_task(drps, components, tmp_path):
    dal = make_dal(drps, components, tmp_path)
    task = dal.new_task("reduce", {"oblock_id": 3})
    assert task.id == 3
    assert task.request == "reduce"
    assert task.request_runinfo["runner"] == "numina"
    assert dal.new_task_id("other", {}) == 1


def search(dal, obsid, query):
    obsres = dal.obsres_from_oblock_id(obsid, profile=PROFILE)
    return dal.search_result_relative("value", int, obsres, query)


@pytest.mark.xfail(reason="an observing block without result stops the search", strict=True)
def test_result_prev_skips_missing(drps, components, tmp_path):
    write_result(tmp_path, 1, {"value": 5})
    dal = make_dal(drps, components, tmp_path)

    assert search(dal, 3, ResultOf("value", node="prev")).content == 5


def test_result_prev(drps, components, tmp_path):
    write_result(tmp_path, 1, {"value": 5})
    write_result(tmp_path, 2, {"value": 6})
    dal = make_dal(drps, components, tmp_path)

    assert search(dal, 3, ResultOf("value", node="prev")).content == 6
    # the mode of the observing block must match
    assert search(dal, 3, ResultOf("bias.value", node="prev")).content == 5


def test_result_prev_not_found(drps, components, tmp_path):
    write_result(tmp_path, 1, {"other": 5})
    write_result(tmp_path, 2, {"other": 6})
    dal = make_dal(drps, components, tmp_path)

    with pytest.raises(NoResultFound, match="value not found in any node"):
        search(dal, 3, ResultOf("value", node="prev"))


def test_result_children(drps, components, tmp_path):
    write_result(tmp_path, 1, {"value": 5})
    write_result(tmp_path, 2, {"value": 6})
    dal = make_dal(drps, components, tmp_path)

    results = search(dal, 10, ResultOf("value", node="children"))

    assert [res.content for res in results] == [5, 6]
    assert [res.id for res in results] == [1, 2]


@pytest.mark.xfail(reason="a child without result raises FileNotFoundError", strict=True)
@pytest.mark.parametrize("ignore_fail", [True, False])
def test_result_children_missing(drps, components, tmp_path, ignore_fail):
    write_result(tmp_path, 1, {"value": 5})
    dal = make_dal(drps, components, tmp_path)
    query = ResultOf("value", node="children", ignore_fail=ignore_fail)

    if ignore_fail:
        assert [res.content for res in search(dal, 10, query)] == [5]
    else:
        with pytest.raises(NoResultFound):
            search(dal, 10, query)


def test_result_children_missing_field(drps, components, tmp_path):
    write_result(tmp_path, 1, {"value": 5})
    write_result(tmp_path, 2, {"other": 6})
    dal = make_dal(drps, components, tmp_path)

    query = ResultOf("value", node="children", ignore_fail=True)
    assert [res.content for res in search(dal, 10, query)] == [5]

    with pytest.raises(NoResultFound, match="no field value found in result"):
        search(dal, 10, ResultOf("value", node="children"))


class UnknownNode:
    mode = None
    attr = "value"
    node = "other"
    ignore_fail = False


def test_result_unknown_node(drps, components, tmp_path):
    dal = make_dal(drps, components, tmp_path)
    with pytest.raises(TypeError, match="unknown node type other"):
        search(dal, 3, UnknownNode())


def test_base_hybrid_not_implemented(drps):
    dal = BaseHybridDAL(drps, "", {}, {})
    with pytest.raises(NotImplementedError):
        dal.search_result_id(1, int, "value")
    with pytest.raises(NotImplementedError):
        dal._search_prod_table("name", MasterBias(), None)


def test_dict2dal_no_op(drps):
    dal = Dict2DAL(drps, {}, {})
    assert dal.update_task(None) is None
    assert dal.update_result(None, None, None) is None


class NamedType:
    """A type whose label ends in ()"""

    @staticmethod
    def name():
        return "DemoType1()"


def test_label_with_parenthesis(drps, components, tmp_path):
    base = {"products": {"TEST1": {PROFILE: [{"id": 4, "type": "DemoType1", "tags": {}, "content": {"a": 1}}]}}}
    dal = make_dal(drps, components, tmp_path, base)
    obsres = dal.obsres_from_oblock_id(2, profile=PROFILE)

    result = dal.search_product("demo", NamedType(), obsres)

    assert result.id == 4
    assert result.content == {"a": 1}


class FakeExtractor:
    def __init__(self, value):
        self.value = value

    def extract(self, key, img):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class FakeDatamodel:
    def __init__(self, value):
        self.extractor_map = {"fits": FakeExtractor(value)}


class FakeDrp:
    def __init__(self, value):
        self.datamodel = FakeDatamodel(value)


class Frame:
    frame = None

    def __init__(self):
        self.closed = False

    def open(self):
        return self

    def close(self):
        self.closed = True


class FakeObsres:
    id = 1

    def __init__(self, frame):
        self.frame = frame

    def get_sample_frame(self):
        return self.frame


@pytest.mark.parametrize("value", ["not a date", 3.0, KeyError("observation_date")])
def test_observation_date_invalid(value):
    frame = Frame()
    assert BaseHybridDAL._observation_date(FakeObsres(frame), FakeDrp(value)) is None


def test_observation_date_closes_file():
    frame = Frame()
    BaseHybridDAL._observation_date(FakeObsres(frame), FakeDrp("2020-01-01T00:00:00"))
    assert frame.closed
