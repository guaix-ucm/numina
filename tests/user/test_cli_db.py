"""numina db: observing blocks in the registry of reductions, reduced by id"""

import json

import pytest
import yaml

from numina.dal.registry import Registry
from numina.user.cli import base_config, main
from numina.user.helpers import create_datamanager
from numina.user.session import Session

from . import test_run_registry as trr

BIAS = trr.BIAS
IMAGE = {"id": 10, "mode": "image", "instrument": "TEST1", "images": ["image2.fits"]}
PARENT = {"id": 20, "mode": "param", "instrument": "TEST1", "children": [1, 10]}


@pytest.fixture
def basedir(drpmocker, tmp_path, monkeypatch):
    basedir = trr.create_basedir(drpmocker, tmp_path, monkeypatch)
    (basedir / "obs.yaml").write_text(yaml.safe_dump_all([BIAS, IMAGE]))
    return basedir


def numina_db(capsys, *args):
    status = main(["--disable-plugins", "db", *args])
    captured = capsys.readouterr()
    return status, captured.out.splitlines(), captured.err


def registry(basedir):
    return Registry(str(basedir / "numina-db.json"), basedir=str(basedir))


def test_ingest(basedir, capsys):
    status, lines, _ = numina_db(capsys, "ingest", "--db", "numina-db.json", "obs.yaml")
    assert status == 0
    assert lines == ["new oblock_id=1", "new oblock_id=10"]
    assert registry(basedir).oblock(10) == IMAGE
    # no reductions
    assert registry(basedir).tasks() == []

    status, lines, _ = numina_db(capsys, "ingest", "--db", "numina-db.json", "obs.yaml")
    assert lines == ["unchanged oblock_id=1", "unchanged oblock_id=10"]

    changed = dict(BIAS, images=["image2.fits"])
    (basedir / "obs.yaml").write_text(yaml.safe_dump(changed))
    status, lines, _ = numina_db(capsys, "ingest", "--db", "numina-db.json", "obs.yaml")
    assert lines == ["updated oblock_id=1"]
    assert registry(basedir).oblock(1) == changed


def test_run_by_id(basedir, capsys):
    numina_db(capsys, "ingest", "--db", "numina-db.json", "obs.yaml")
    status, _, _ = numina_db(capsys, "run", "--db", "numina-db.json", "1", "10")
    assert status == 0

    tasks = registry(basedir).tasks()
    assert [(task["oblock_id"], task["state"]) for task in tasks] == [(1, 2), (10, 2)]
    # the image used the master bias of the first reduction
    (product,) = registry(basedir).products(type="MasterBias")
    assert (basedir / product["content"]).is_file()
    assert len(trr.SEEN) == 1


def test_run_unknown_id(basedir, capsys):
    numina_db(capsys, "ingest", "--db", "numina-db.json", "obs.yaml")
    status, _, err = numina_db(capsys, "run", "--db", "numina-db.json", "1", "99")
    assert status == 2
    assert "oblock_id=99 is not in the registry" in err
    # nothing is reduced
    assert registry(basedir).tasks() == []


@pytest.mark.parametrize(
    "args", [["ingest", "obs.yaml"], ["run", "1"], ["list", "oblocks"]], ids=["ingest", "run", "list"]
)
def test_no_registry(basedir, capsys, monkeypatch, args):
    # without the configuration files of the user
    monkeypatch.setattr("numina.user.cli.load_config", base_config)
    status, _, err = numina_db(capsys, *args)
    assert status == 2
    assert "no registry of reductions" in err


def test_list(basedir, capsys):
    numina_db(capsys, "ingest", "--db", "numina-db.json", "obs.yaml")
    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "oblocks")
    assert status == 0
    assert lines[0].split() == ["oblock_id", "instrument", "mode", "frames"]
    assert [line.split() for line in lines[1:]] == [["1", "TEST1", "bias", "1"], ["10", "TEST1", "image", "1"]]

    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "oblocks", "--mode", "image")
    assert [line.split()[0] for line in lines[1:]] == ["10"]

    numina_db(capsys, "run", "--db", "numina-db.json", "1")
    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "tasks")
    assert lines[1].split()[:5] == ["1", "1", "TEST1", "bias", "finished"]
    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "products")
    assert lines[0].split()[:4] == ["id", "uuid", "type", "oblock_id"]
    product_id, _, product_type, oblock_id = lines[1].split()[:4]
    assert (product_id, product_type, oblock_id) == ("1", "MasterBias", "1")

    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "results")
    assert lines[0].split()[:3] == ["id", "uuid", "task_id"]
    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "results", "--oblock-id", "10")
    assert lines == ["no results"]


def test_list_unknown_field(basedir, capsys):
    numina_db(capsys, "ingest", "--db", "numina-db.json", "obs.yaml")
    status, _, err = numina_db(capsys, "list", "--db", "numina-db.json", "products", "--mode", "bias")
    assert status == 2
    assert "the products have no field mode" in err


def test_session_runs_ids_of_the_registry(basedir, capsys):
    numina_db(capsys, "ingest", "--db", "numina-db.json", "obs.yaml")
    session = Session(basedir=basedir, config=base_config(), db="numina-db.json")
    # the blocks are not added to the session, they are read from the registry
    assert session.run(1).state == 2
    assert session.run(10).state == 2
    assert len(trr.SEEN) == 1


def test_session_unknown_id(basedir):
    session = Session(basedir=basedir, config=base_config(), db="numina-db.json")
    with pytest.raises(ValueError, match="oblock_id=1 is not in the observation results or the registry"):
        session.run(1)


def test_restore_children(basedir):
    reg = registry(basedir)
    for oblock in [BIAS, IMAGE, PARENT]:
        reg.record_oblock(oblock)
    config = base_config()
    config["tool.run"]["basedir"] = str(basedir)
    config["tool.db"]["file"] = "numina-db.json"
    datamanager = create_datamanager(config, None)

    datamanager.restore_oblock(20)
    assert set(datamanager.backend.ob_table) == {1, 10, 20}
    assert datamanager.backend.ob_table[1]["parent"] == 20


def test_loaded_oblock_has_precedence(basedir):
    """A block loaded from a file is not replaced by the registry"""
    reg = registry(basedir)
    reg.record_oblock(dict(BIAS, images=["image2.fits"]))
    session = Session(basedir=basedir, config=base_config(), db="numina-db.json")
    session.add_observations(BIAS)
    session.run(1)
    # the definition reduced is recorded in the registry
    assert registry(basedir).oblock(1) == BIAS


def test_priority(basedir, capsys):
    numina_db(capsys, "ingest", "--db", "numina-db.json", "obs.yaml")
    numina_db(capsys, "run", "--db", "numina-db.json", "1")
    numina_db(capsys, "run", "--db", "numina-db.json", "1")
    first, second = registry(basedir).products(type="MasterBias")

    # without priorities, the most recent
    numina_db(capsys, "run", "--db", "numina-db.json", "10")
    assert trr.SEEN[-1] == second["uuid"]

    status, lines, _ = numina_db(capsys, "priority", "--db", "numina-db.json", str(first["id"]), "2")
    assert status == 0
    assert lines == [f"product {first['id']} has priority 2"]
    numina_db(capsys, "run", "--db", "numina-db.json", "10")
    assert trr.SEEN[-1] == first["uuid"]

    # by uuid, abbreviated
    status, _, _ = numina_db(capsys, "priority", "--db", "numina-db.json", second["uuid"][:8], "3")
    assert status == 0
    numina_db(capsys, "run", "--db", "numina-db.json", "10")
    assert trr.SEEN[-1] == second["uuid"]

    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "products")
    header = lines[0].split()
    priorities = [line.split()[header.index("priority")] for line in lines[1:]]
    assert priorities == ["2", "3"]


def test_priority_unknown_product(basedir, capsys):
    numina_db(capsys, "ingest", "--db", "numina-db.json", "obs.yaml")
    status, _, err = numina_db(capsys, "priority", "--db", "numina-db.json", "99", "1")
    assert status == 2
    assert "no product with id 99" in err


def test_session_set_priority(basedir):
    session = Session(basedir=basedir, config=base_config(), db="numina-db.json")
    session.add_observations("obs.yaml")
    session.run(1)
    session.run(1)
    first, _ = session.products(type="MasterBias")
    assert session.set_priority(first["uuid"], 1) == first["id"]
    session.run(10)
    assert trr.SEEN[-1] == first["uuid"]


@pytest.fixture
def two_biases(basedir, capsys):
    numina_db(capsys, "ingest", "--db", "numina-db.json", "obs.yaml")
    numina_db(capsys, "run", "--db", "numina-db.json", "1")
    numina_db(capsys, "run", "--db", "numina-db.json", "1")
    reg = registry(basedir)
    first, second = reg.products(type="MasterBias")
    # tags of the products, the test recipe produces none
    with reg.store.transaction():
        reg.store.update("products", first["id"], {"tags": {"vph": "A", "mode": "X"}})
        reg.store.update("products", second["id"], {"tags": {"vph": "B"}})
    return reg.products(type="MasterBias")


def test_list_table_abbreviates(two_biases, capsys):
    first, _ = two_biases
    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "products")
    header = lines[0].split()
    row = lines[1].split()
    assert row[header.index("uuid")] == first["uuid"][:8]
    assert row[header.index("time_obs")] == first["time_obs"][:19]
    assert len(lines) == 3


def test_list_tags(two_biases, capsys):
    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "products", "--tags")
    assert status == 0
    assert lines[2] == "    └─ mode=X vph=A"
    assert lines[4] == "    └─ vph=B"


def test_list_tags_none(basedir, capsys):
    numina_db(capsys, "ingest", "--db", "numina-db.json", "obs.yaml")
    numina_db(capsys, "run", "--db", "numina-db.json", "1")
    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "products", "--tags")
    assert lines[2] == "    └─ (no tags)"


def test_list_tag_filter(two_biases, capsys):
    first, second = two_biases
    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "products", "--tag", "vph=B")
    assert [line.split()[0] for line in lines[1:]] == [str(second["id"])]
    status, lines, _ = numina_db(
        capsys, "list", "--db", "numina-db.json", "products", "--tag", "vph=A", "--tag", "mode=X"
    )
    assert [line.split()[0] for line in lines[1:]] == [str(first["id"])]
    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "products", "--tag", "mode=Y")
    assert lines == ["no products"]


def test_list_json(two_biases, capsys):
    first, second = two_biases
    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "products", "--json", "--tag", "vph=A")
    assert status == 0
    (doc,) = json.loads("\n".join(lines))
    assert doc["uuid"] == first["uuid"]
    assert doc["tags"] == {"vph": "A", "mode": "X"}

    status, lines, _ = numina_db(capsys, "list", "--db", "numina-db.json", "tasks", "--json")
    assert [doc["oblock_id"] for doc in json.loads("\n".join(lines))] == [1, 1]


@pytest.mark.parametrize(
    "args, match",
    [
        (["tasks", "--tags"], "only the products have tags"),
        (["oblocks", "--tag", "vph=A"], "only the products have tags"),
        (["products", "--tag", "vph"], "the tag vph is not KEY=VALUE"),
    ],
)
def test_list_tags_errors(two_biases, capsys, args, match):
    status, _, err = numina_db(capsys, "list", "--db", "numina-db.json", *args)
    assert status == 2
    assert match in err


def test_list_tags_and_json_not_allowed(basedir):
    with pytest.raises(SystemExit):
        main(["--disable-plugins", "db", "list", "--db", "numina-db.json", "products", "--tags", "--json"])
