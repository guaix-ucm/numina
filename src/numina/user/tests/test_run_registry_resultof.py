"""ResultOf with a registry of reductions"""

import argparse
import os

import astropy.io.fits as fits
import pytest
import yaml

from numina.dal.registry import Registry
from numina.exceptions import NoResultFound

from ..cli import base_config, process_unknown_arguments
from ..clirun import register
from ..clirundal import mode_run_common_obs
from .test_nested_profile import DRP_TEST1

CHILDREN = [
    {"id": 1, "mode": "child", "instrument": "TEST1", "images": ["image1.fits"]},
    {"id": 2, "mode": "child", "instrument": "TEST1", "images": ["image2.fits"]},
]
PARENT = {"id": 3, "mode": "parent", "instrument": "TEST1", "children": [1, 2]}


@pytest.fixture
def basedir(drpmocker, tmp_path, monkeypatch):
    drpmocker.add_drp("TEST1", DRP_TEST1)
    datadir = tmp_path / "data"
    datadir.mkdir()
    for name in ["image1.fits", "image2.fits"]:
        hdr = fits.Header()
        hdr["INSTRUME"] = "TEST1"
        hdr["DATE-OBS"] = "2015-01-01T00:00:00"
        fits.PrimaryHDU(header=hdr).writeto(datadir / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(basedir, obs, *options, db=True):
    obsfile = basedir / "obsdata.yaml"
    obsfile.write_text(yaml.safe_dump_all(obs))
    parser = argparse.ArgumentParser(prog="numina")
    register(parser.add_subparsers(), base_config())
    db_options = ["--db", "numina-db.json"] if db else []
    args = parser.parse_args(["run", *db_options, *options, str(obsfile)])
    config = base_config()
    config["tool.run"]["basedir"] = str(basedir)
    mode_run_common_obs(args, process_unknown_arguments([]), config)


def disabled(obs):
    return [dict(ob, enabled=False) for ob in obs]


def test_results_of_previous_run(basedir):
    """The parent finds the results of the children reduced in a previous run"""
    run(basedir, CHILDREN)
    run(basedir, disabled(CHILDREN) + [PARENT])

    registry = Registry(str(basedir / "numina-db.json"))
    states = {task["oblock_id"]: task["state"] for task in registry.tasks()}
    assert states == {1: 2, 2: 2, 3: 2}


def test_most_recent_result(basedir, monkeypatch):
    """The most recent result of each child is used"""
    import numina.dal.dictdal as dictdal

    run(basedir, CHILDREN)
    run(basedir, CHILDREN)
    used = []
    original = dictdal.HybridDAL._load_result_field

    def recording(self, node_id, directory, filename, field):
        used.append(directory)
        return original(self, node_id, directory, filename, field)

    monkeypatch.setattr(dictdal.HybridDAL, "_load_result_field", recording)
    run(basedir, disabled(CHILDREN) + [PARENT])
    assert used == ["obsid1_3_results", "obsid2_4_results"]


def test_no_result_in_registry(basedir):
    """Without results in the registry, the directory can not be built from the templates"""
    with pytest.raises(NoResultFound, match="no result of oblock_id=1 in the registry"):
        run(basedir, disabled(CHILDREN) + [PARENT])


@pytest.mark.parametrize("db", [False, True])
def test_missing_child(basedir, db):
    """A child that is not defined is an error"""
    with pytest.raises(ValueError, match="oblock_id=1, child of oblock_id=3, not found"):
        run(basedir, [PARENT], db=db)


@pytest.mark.parametrize("db", [False, True])
def test_relative_basedir(basedir, db):
    """With a relative --basedir, datadir and the results are inside basedir"""
    workdir = basedir / "base"
    workdir.mkdir()
    (basedir / "data").rename(workdir / "data")
    run(basedir, CHILDREN + [PARENT], "--basedir", "base", db=db)

    for obsid in [1, 2, 3]:
        assert any(name.startswith(f"obsid{obsid}_") for name in os.listdir(workdir))
    assert not [name for name in os.listdir(basedir) if name.startswith("obsid")]
