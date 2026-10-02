"""numina run with a control file in format 2 (persistent database)"""

import argparse
import os

import astropy.io.fits as fits
import pytest
import yaml

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
def workdir(drpmocker, tmp_path, monkeypatch):
    """basedir with data, the current directory is basedir"""
    drpmocker.add_drp("TEST1", DRP_TEST1)
    basedir = tmp_path / "base"
    datadir = basedir / "data"
    datadir.mkdir(parents=True)
    monkeypatch.chdir(basedir)
    for name in ["image1.fits", "image2.fits"]:
        hdr = fits.Header()
        hdr["INSTRUME"] = "TEST1"
        hdr["DATE-OBS"] = "2015-01-01T00:00:00"
        fits.PrimaryHDU(header=hdr).writeto(datadir / name)
    return basedir


def write_control(basedir, version=2):
    control = basedir / "control.yaml"
    if version == 2:
        control.write_text("version: 2\ndatabase: {}\n")
    else:
        control.write_text("version: 1\n")
    return control


def run(basedir, obs, *options):
    obsfile = basedir / "obsdata.yaml"
    obsfile.write_text(yaml.safe_dump_all(obs))
    parser = argparse.ArgumentParser(prog="numina")
    register(parser.add_subparsers(), base_config())
    args = parser.parse_args(["run", *options, str(obsfile)])
    config = base_config()
    config["tool.run"]["basedir"] = str(basedir)
    mode_run_common_obs(args, process_unknown_arguments([]), config)


def load_database(control):
    with open(control) as fd:
        return yaml.safe_load(fd)["database"]


def test_results_of_previous_runs(workdir):
    """The parent OB finds the results of the children stored in a previous run"""
    control = write_control(workdir)
    run(workdir, CHILDREN, "-r", str(control))
    run(workdir, CHILDREN + [PARENT], "-e", "3", "-r", str(control))

    database = load_database(control)
    states = {task["request_params"]["oblock_id"]: task["state"] for task in database["tasks"].values()}
    assert states == {1: 2, 2: 2, 3: 2}


def test_failed_task_is_stored(workdir):
    """A task that fails before running the recipe is stored as failed"""
    control = write_control(workdir)
    # the children are not reduced, so they have no results
    children = [dict(ob, enabled=False) for ob in CHILDREN]
    with pytest.raises(NoResultFound):
        run(workdir, children + [PARENT], "-r", str(control))

    database = load_database(control)
    (task,) = database["tasks"].values()
    assert task["state"] == 3
    assert task["time_start"] is None
    assert task["time_end"] is None


def test_no_control_dump(workdir):
    """The database is not copied to control_dump.json in the current directory"""
    control = write_control(workdir)
    run(workdir, CHILDREN, "-r", str(control))

    assert not (workdir / "control_dump.json").exists()
    # no temporary files left
    assert sorted(os.listdir(workdir)) == sorted(
        ["control.yaml", "data", "obsdata.yaml", "obsid1_1_result", "obsid1_1_work", "obsid2_2_result", "obsid2_2_work"]
    )


def test_database_not_modified_on_error(workdir, monkeypatch):
    """An error writing the database leaves the previous version"""
    from numina.dal.backend import Backend

    control = write_control(workdir)
    run(workdir, CHILDREN[:1], "-r", str(control))
    before = control.read_text()

    def failing_dump(self, fp):
        fp.write("partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(Backend, "dump", failing_dump)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(workdir, CHILDREN, "-r", str(control))

    assert control.read_text() == before
    assert not [name for name in os.listdir(workdir) if name.endswith(".tmp")]


def test_missing_child(workdir):
    """A child that is not defined is an error"""
    control = write_control(workdir)
    with pytest.raises(ValueError, match="oblock_id=1, child of oblock_id=3, not found"):
        run(workdir, [PARENT], "-r", str(control))


@pytest.mark.parametrize("version", [1, 2])
def test_relative_basedir(workdir, tmp_path, monkeypatch, version):
    """With a relative --basedir, datadir and the results are inside basedir"""
    monkeypatch.chdir(tmp_path)
    control = write_control(workdir, version)
    run(workdir, CHILDREN + [PARENT], "-r", str(control), "--basedir", "base")

    for obsid in [1, 2, 3]:
        assert any(name.startswith(f"obsid{obsid}_") for name in os.listdir(workdir))
    assert not [name for name in os.listdir(tmp_path) if name.startswith("obsid")]
