"""numina db: observing blocks in the registry of reductions, reduced by id"""

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
    assert lines[1].split()[:3] == ["1", "MasterBias", "1"]

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
