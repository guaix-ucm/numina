"""numina run with a registry of reductions (--db)"""

import argparse
import logging
import uuid

import astropy.io.fits as fits
import pytest
import yaml

from numina.core import BaseRecipe, Result
from numina.core.requirements import ObservationResultRequirement
from numina.dal.registry import Registry
from numina.tests.recipes import MasterBias

from ..cli import base_config, process_unknown_arguments
from ..clirun import register
from ..clirundal import mode_run_common_obs

DRP_TEST1 = """
name: TEST1
configurations:
  path: numina.drps.tests.configs
modes:
  - key: bias
    name: Bias
    summary: Bias mode
    description: Bias mode
  - key: fail
    name: Fail
    summary: Fail mode
    description: Fail mode
pipelines:
  default:
    version: 1
    recipes:
      bias: numina.user.tests.test_run_registry.BiasRecipe
      fail: numina.core.utils.AlwaysFailRecipe
"""


class BiasRecipe(BaseRecipe):
    obresult = ObservationResultRequirement()
    master_bias = Result(MasterBias)

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        with recipe_input.obresult.frames[0].open() as hdul:
            hdr = hdul[0].header.copy()
        hdr["UUID"] = str(uuid.uuid4())
        return self.create_result(master_bias=fits.HDUList([fits.PrimaryHDU(header=hdr)]))


BIAS = {"id": 1, "mode": "bias", "instrument": "TEST1", "images": ["image1.fits"]}


@pytest.fixture
def basedir(drpmocker, tmp_path, monkeypatch):
    drpmocker.add_drp("TEST1", DRP_TEST1)
    datadir = tmp_path / "data"
    datadir.mkdir()
    for name in ["image1.fits", "image2.fits"]:
        hdr = fits.Header()
        hdr["INSTRUME"] = "TEST1"
        hdr["DATE-OBS"] = "2017-01-01T00:00:00"
        fits.PrimaryHDU(header=hdr).writeto(datadir / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(basedir, obs, *options, run_values=None, db_values=None):
    obsfile = basedir / "obsdata.yaml"
    obsfile.write_text(yaml.safe_dump_all(obs))
    parser = argparse.ArgumentParser(prog="numina")
    register(parser.add_subparsers(), base_config())
    args = parser.parse_args(["run", *options, str(obsfile)])
    config = base_config()
    config["tool.run"]["basedir"] = str(basedir)
    for key, value in (run_values or {}).items():
        config["tool.run"][key] = value
    for key, value in (db_values or {}).items():
        config["tool.db"][key] = value
    mode_run_common_obs(args, process_unknown_arguments([]), config)


def test_no_registry(basedir):
    run(basedir, [BIAS])
    assert not list(basedir.glob("*.json"))
    # the default templates
    assert (basedir / "obsid1_results").is_dir()


def test_default_templates_with_registry(basedir):
    """With a registry, the templates of [tool.db] use the id of the task"""
    run(basedir, [BIAS], "--db", "numina-db.json")
    assert (basedir / "obsid1_1_work").is_dir()
    assert (basedir / "obsid1_1_results").is_dir()
    assert not (basedir / "obsid1_results").exists()


def test_user_templates_with_registry(basedir):
    """The templates of [tool.db] can be changed"""
    db_values = {"workdir_tmpl": "work{taskid}", "resultdir_tmpl": "res{taskid}"}
    run(basedir, [BIAS], "--db", "numina-db.json", db_values=db_values)
    assert (basedir / "work1").is_dir()
    assert (basedir / "res1").is_dir()


def test_run_templates_with_registry(basedir):
    """With a registry, the templates of [tool.run] defined in [tool.db] are not used"""
    run_values = {"workdir_tmpl": "work{taskid}", "resultfile_tmpl": "res.json"}
    run(basedir, [BIAS], "--db", "numina-db.json", run_values=run_values)
    assert (basedir / "obsid1_1_work").is_dir()
    # not defined in [tool.db]
    assert (basedir / "obsid1_1_results" / "res.json").is_file()


def test_registry(basedir):
    run(basedir, [BIAS], "--db", "numina-db.json")

    registry = Registry(str(basedir / "numina-db.json"))
    (task,) = registry.tasks()
    assert task["id"] == 1
    assert task["oblock_id"] == 1
    assert task["oblock"] == BIAS
    assert task["state"] == 2
    assert task["time_start"] is not None

    (oblock,) = registry.oblocks()
    assert oblock["definition"] == BIAS

    (result,) = registry.results()
    assert result["task_id"] == 1
    assert result["qc"] == "UNKNOWN"
    assert result["result_dir"] == "obsid1_1_results"

    (product,) = registry.products()
    assert product["origin"] == "reduction"
    assert product["type"] == "MasterBias"
    assert product["instrument"] == "TEST1"
    assert product["result_id"] == result["id"]
    assert product["content"] == "obsid1_1_results/master_bias.fits"
    assert (basedir / product["content"]).is_file()
    assert product["uuid"] == fits.getheader(basedir / product["content"])["UUID"]


def test_registry_in_config(basedir):
    """The registry can be defined in [tool.db], relative to basedir"""
    (basedir / "sub").mkdir()
    run(basedir, [BIAS], db_values={"file": "sub/numina-db.json"})
    assert len(Registry(str(basedir / "sub" / "numina-db.json")).tasks()) == 1


def test_several_runs(basedir):
    """The tasks and the products accumulate, the OB is updated"""
    run(basedir, [BIAS], "--db", "numina-db.json")
    changed = dict(BIAS, images=["image2.fits"])
    run(basedir, [changed], "--db", "numina-db.json")

    registry = Registry(str(basedir / "numina-db.json"))
    tasks = registry.tasks()
    assert [task["id"] for task in tasks] == [1, 2]
    # each task keeps the OB it processed
    assert tasks[0]["oblock"]["images"] == ["image1.fits"]
    assert tasks[1]["oblock"]["images"] == ["image2.fits"]
    (oblock,) = registry.oblocks()
    assert oblock["definition"]["images"] == ["image2.fits"]

    products = registry.products(type="MasterBias")
    assert [p["content"] for p in products] == [
        "obsid1_1_results/master_bias.fits",
        "obsid1_2_results/master_bias.fits",
    ]


def test_failed_task(basedir):
    fail = {"id": 2, "mode": "fail", "instrument": "TEST1", "images": ["image1.fits"]}
    with pytest.raises(TypeError, match="This Recipe always fails"):
        run(basedir, [fail], "--db", "numina-db.json")

    registry = Registry(str(basedir / "numina-db.json"))
    (task,) = registry.tasks()
    assert task["state"] == 3
    assert registry.results() == []


def test_templates_without_taskid(basedir, caplog):
    caplog.set_level(logging.WARNING)
    run(basedir, [BIAS], "--db", "numina-db.json", db_values={"resultdir_tmpl": "results_{obsid}"})
    assert "resultdir_tmpl=results_{obsid} does not use {taskid}" in caplog.text


def test_format2_with_registry(basedir):
    control = basedir / "control.yaml"
    control.write_text("version: 2\ndatabase: {}\n")
    with pytest.raises(ValueError, match="requires a control file in format 1"):
        run(basedir, [BIAS], "--db", "numina-db.json", "-r", str(control))
