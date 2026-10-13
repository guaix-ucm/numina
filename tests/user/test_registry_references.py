"""References to the registry in the requirements: {product: ...}, {result_of: ...}, {task: ...}"""

import json
import os

import pytest
import yaml

from numina.dal.registry import Registry, is_reference
from numina.exceptions import ValidationError
from numina.user.cli import base_config
from numina.user.baserun import check_reference_tags
from numina.user.session import Session

from . import test_run_registry as trr

BIAS = trr.BIAS
IMAGE = {"id": 10, "mode": "image", "instrument": "TEST1", "images": ["image2.fits"]}


@pytest.fixture
def session(drpmocker, tmp_path, monkeypatch):
    """A session with a registry and two master bias, of tasks 1 and 2"""
    basedir = trr.create_basedir(drpmocker, tmp_path, monkeypatch)
    (basedir / "obs.yaml").write_text(yaml.safe_dump_all([BIAS, IMAGE]))
    session = Session(basedir=basedir, config=base_config(), db="numina-db.json")
    session.add_observations("obs.yaml")
    session.run(1)
    session.run(1)
    return session


def last_inputs(session):
    task = session.tasks()[-1]
    return task["request_runinfo"]["inputs"]


def test_is_reference():
    assert is_reference({"product": 1})
    assert is_reference({"result_of": "bias", "field": "master_bias"})
    assert not is_reference({"value": 1})
    assert not is_reference("master_bias.fits")


def test_automatic_selection(session):
    """Without reference, the most recent master bias"""
    first, second = session.products(type="MasterBias")
    session.run(10)
    assert trr.SEEN[-1] == second["uuid"]
    assert last_inputs(session)["master_bias"]["product_id"] == second["id"]


@pytest.mark.parametrize("by", ["id", "uuid"])
def test_product_reference(session, by):
    first, _ = session.products(type="MasterBias")
    key = first["id"] if by == "id" else first["uuid"]
    session.run(10, requirements={"master_bias": {"product": key}})
    assert trr.SEEN[-1] == first["uuid"]
    inputs = last_inputs(session)["master_bias"]
    assert inputs["source"] == "registry"
    assert inputs["product_id"] == first["id"]
    assert inputs["reference"] == {"product": key}
    assert inputs["requested_in"] == "request"


def test_result_of_reference(session):
    _, second = session.products(type="MasterBias")
    session.run(10, requirements={"master_bias": {"result_of": 1}})
    assert trr.SEEN[-1] == second["uuid"]
    # the id can be a string
    session.run(10, requirements={"master_bias": {"result_of": "1", "field": "master_bias"}})
    assert trr.SEEN[-1] == second["uuid"]


def test_task_reference(session):
    first, _ = session.products(type="MasterBias")
    session.run(10, requirements={"master_bias": {"task": first["task_id"]}})
    assert trr.SEEN[-1] == first["uuid"]
    assert last_inputs(session)["master_bias"]["result_id"] == first["result_id"]


def test_reference_in_observing_block(session):
    first, _ = session.products(type="MasterBias")
    image = dict(IMAGE, id=11, requirements={"master_bias": {"product": first["id"]}})
    session.add_observations(image)
    session.run(11)
    assert trr.SEEN[-1] == first["uuid"]
    assert last_inputs(session)["master_bias"]["requested_in"] == "observation block"


@pytest.mark.parametrize(
    "reference, match",
    [
        ({"product": 99}, "no product with id 99 in the registry"),
        ({"product": 12345678}, "for a product with uuid 12345678 use {product: '12345678'}, with quotes"),
        ({"product": "no-uuid"}, "no product with uuid no-uuid in the registry"),
        ({"product": "00000000-0000"}, "no product with uuid 00000000-0000 in the registry"),
        ({"product": "abc"}, "an abbreviated uuid needs at least 8 characters"),
        ({"product": "1"}, "for the product with id 1 use {product: 1}, without quotes"),
        ({"result_of": 99}, "no result of oblock_id=99"),
        ({"task": 99}, "task 99 is not in the registry"),
        ({"result": 99}, "no result with id 99 in the registry"),
        ({"result": "abc"}, "an abbreviated uuid needs at least 8 characters"),
        ({"result_of": 1, "field": "other"}, "has no field other"),
        ({"product": 1, "field": "master_bias"}, "a product has no field"),
        ({"product": 1, "task": 1}, "invalid reference"),
        ({"product": 1, "other": 1}, "invalid reference"),
    ],
)
def test_invalid_reference(session, reference, match):
    with pytest.raises(ValidationError, match=f"requirement master_bias: .*{match}"):
        session.run(10, requirements={"master_bias": reference})
    assert session.tasks()[-1]["state"] == 3


def test_reference_without_registry(drpmocker, tmp_path, monkeypatch):
    basedir = trr.create_basedir(drpmocker, tmp_path, monkeypatch)
    session = Session(basedir=basedir, config=base_config())
    session.add_observations(IMAGE)
    with pytest.raises(ValidationError, match="there is no registry of reductions"):
        session.run(10, requirements={"master_bias": {"product": 1}})


def test_resolve_reference(session):
    registry = Registry(session.registry.filename, basedir=session.basedir)
    first, _ = session.products(type="MasterBias")
    value, details = registry.resolve_reference({"product": first["id"]}, "master_bias")
    assert value == os.path.join(session.basedir, first["content"])
    assert details == {"product_id": first["id"], "tags": {}}


def add_other_product(session):
    """A product of other type, in the registry"""
    first, _ = session.products(type="MasterBias")
    doc = dict(first, type="TraceMap", type_fqn="numina.types.TraceMap")
    del doc["id"]
    with session.registry.store.transaction():
        return session.registry.store.insert("products", doc)


def test_reference_of_other_type(session):
    other = add_other_product(session)
    with pytest.raises(ValidationError, match=f"the product {other} is a TraceMap, not a MasterBias"):
        session.run(10, requirements={"master_bias": {"product": other}})


def test_result_field_of_other_type(session):
    first, _ = session.products(type="MasterBias")
    with session.registry.store.transaction():
        session.registry.store.update("products", first["id"], {"type": "TraceMap"})
    with pytest.raises(ValidationError, match="the field master_bias of the result .* is a TraceMap, not a MasterBias"):
        session.run(10, requirements={"master_bias": {"task": first["task_id"]}})


def test_reference_of_unknown_requirement(session):
    """The references of requirements that the recipe does not have are not checked"""
    other = add_other_product(session)
    session.run(10, requirements={"other": {"product": other}}, strict_inputs=False)
    assert session.tasks()[-1]["state"] == 2


def test_abbreviated_uuid(session):
    first, _ = session.products(type="MasterBias")
    session.run(10, requirements={"master_bias": {"product": first["uuid"][:8]}})
    assert trr.SEEN[-1] == first["uuid"]
    assert last_inputs(session)["master_bias"]["product_id"] == first["id"]


def test_ambiguous_uuid(session):
    first, second = session.products(type="MasterBias")
    prefix = "abcdef01"
    with session.registry.store.transaction():
        for doc, end in [(first, "-1"), (second, "-2")]:
            session.registry.store.update("products", doc["id"], {"uuid": prefix + end})
    with pytest.raises(ValidationError, match=f"the uuid {prefix} is ambiguous, .* of the products 1, 2"):
        session.run(10, requirements={"master_bias": {"product": prefix}})
    # a longer prefix is not ambiguous
    session.run(10, requirements={"master_bias": {"product": prefix + "-2"}})
    assert last_inputs(session)["master_bias"]["product_id"] == second["id"]


@pytest.mark.parametrize(
    "stored, expected",
    [
        ("master_bias.fits", "path"),  # a file of the result
        ("LCB", "LCB"),  # a string value
        (100.0, 100.0),
        ([1, 2], [1, 2]),
    ],
)
def test_result_values(session, stored, expected):
    """The strings of the result are files only if they are files of the result"""
    first, _ = session.products(type="MasterBias")
    result = session.registry.store.get("results", first["result_id"])
    result_file = os.path.join(session.basedir, result["result_dir"], result["result_file"])
    with open(result_file) as fd:
        data = json.load(fd)
    data["values"]["other"] = stored
    with open(result_file, "w") as fd:
        json.dump(data, fd)

    value, _ = session.registry.resolve_reference({"task": first["task_id"], "field": "other"}, "x")
    if expected == "path":
        assert value == os.path.join(session.basedir, result["result_dir"], stored)
    else:
        assert value == expected


@pytest.mark.parametrize("by", ["id", "uuid", "prefix"])
def test_result_reference(session, by):
    first, _ = session.products(type="MasterBias")
    result = session.registry.store.get("results", first["result_id"])
    key = {"id": result["id"], "uuid": result["uuid"], "prefix": result["uuid"][:8]}[by]
    session.run(10, requirements={"master_bias": {"result": key}})
    assert trr.SEEN[-1] == first["uuid"]
    assert last_inputs(session)["master_bias"]["result_id"] == result["id"]


def test_task_failed(session):
    session.add_observations({"id": 40, "mode": "fail", "instrument": "TEST1"})
    with pytest.raises(TypeError, match="This Recipe always fails"):
        session.run(40)
    (failed,) = [task for task in session.tasks() if task["oblock_id"] == 40]
    with pytest.raises(ValidationError, match=f"task {failed['id']} failed, it has no result"):
        session.run(10, requirements={"master_bias": {"task": failed["id"]}})


def test_bad_result_warns(session, caplog):
    first, _ = session.products(type="MasterBias")
    with session.registry.store.transaction():
        session.registry.store.update("results", first["result_id"], {"qc": "BAD"})
    with caplog.at_level("WARNING", logger="numina"):
        session.run(10, requirements={"master_bias": {"task": first["task_id"]}})
    assert f"the result {first['result_id']} has QC BAD" in caplog.text
    # an explicit reference is used
    assert trr.SEEN[-1] == first["uuid"]


class TaggedType:
    """A product type with the tags vph and insmode"""

    def __init__(self):
        from numina.core.tagexpr import query_expr_from_attr
        from numina.keydef import QueryAttribute

        self.query_expr = query_expr_from_attr([QueryAttribute("vph", str), QueryAttribute("insmode", str)])

    def isproduct(self):
        return True


class TaggedRecipe:
    def __init__(self):
        from types import SimpleNamespace

        self._reqs = {"master_traces": SimpleNamespace(dest="master_traces", alias="traces", type=TaggedType())}

    def requirements(self):
        return self._reqs


def tagged_obsres(**tags):
    from types import SimpleNamespace

    return SimpleNamespace(tags=tags)


@pytest.mark.parametrize("key", ["master_traces", "traces"])
def test_reference_tags_match(key):
    resolved = {key: ("the product 3", {"vph": "HR-R", "insmode": "LCB", "temp": 15.9})}
    check_reference_tags(TaggedRecipe(), tagged_obsres(vph="HR-R", insmode="LCB"), resolved)


@pytest.mark.parametrize(
    "tags, match",
    [
        ({"vph": "LR-B", "insmode": "LCB"}, "the product 3 has vph=LR-B, the observing block has vph=HR-R"),
        ({"insmode": "LCB"}, "the product 3 has no tag vph, the observing block has vph=HR-R"),
        (
            {"vph": "LR-B", "insmode": "MOS"},
            "the product 3 has insmode=MOS, vph=LR-B, the observing block has insmode=LCB, vph=HR-R",
        ),
    ],
)
def test_reference_tags_differ(tags, match):
    resolved = {"master_traces": ("the product 3", tags)}
    with pytest.raises(ValidationError, match=f"requirement master_traces: {match}"):
        check_reference_tags(TaggedRecipe(), tagged_obsres(vph="HR-R", insmode="LCB"), resolved)


def test_reference_tags_unknown_requirement():
    resolved = {"other": ("the product 3", {"vph": "LR-B"})}
    check_reference_tags(TaggedRecipe(), tagged_obsres(vph="HR-R", insmode="LCB"), resolved)


def test_reference_tags_recorded(session):
    first, _ = session.products(type="MasterBias")
    session.run(10, requirements={"master_bias": {"product": first["id"]}})
    assert last_inputs(session)["master_bias"]["tags"] == {}
