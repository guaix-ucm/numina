"""numina run with a registry of reductions (--db)"""

import argparse
import logging
import uuid

import astropy.io.fits as fits
import pytest
import yaml

from numina.core import BaseRecipe, Requirement, Result
from numina.core.requirements import ObservationResultRequirement
from numina.dal.registry import Registry
from numina.tests.recipes import MasterBias
from numina.types.qc import QC
import numina.core.dataholders as dh

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
  - key: image
    name: Image
    summary: Image mode
    description: Image mode
  - key: param
    name: Param
    summary: Param mode
    description: Param mode
pipelines:
  default:
    version: 1
    recipes:
      bias: numina.user.tests.test_run_registry.BiasRecipe
      fail: numina.core.utils.AlwaysFailRecipe
      image: numina.user.tests.test_run_registry.ImageRecipe
      param: numina.user.tests.test_run_registry.ParamRecipe
"""

# values received by the recipes
SEEN = []


class BiasRecipe(BaseRecipe):
    obresult = ObservationResultRequirement()
    quality = dh.Parameter("GOOD", "QC of the result")
    master_bias = Result(MasterBias)

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        with recipe_input.obresult.frames[0].open() as hdul:
            hdr = hdul[0].header.copy()
        hdr["UUID"] = str(uuid.uuid4())
        master_bias = fits.HDUList([fits.PrimaryHDU(header=hdr)])
        return self.create_result(master_bias=master_bias, qc=QC[recipe_input.quality])


class ImageRecipe(BaseRecipe):
    obresult = ObservationResultRequirement()
    master_bias = Requirement(MasterBias, "Master bias")

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        with recipe_input.master_bias.open() as hdul:
            SEEN.append(hdul[0].header["UUID"])
        return self.create_result()


class ParamRecipe(BaseRecipe):
    value = dh.Parameter(1, "a value")

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        SEEN.append(recipe_input.value)
        return self.create_result()


BIAS = {"id": 1, "mode": "bias", "instrument": "TEST1", "images": ["image1.fits"]}


@pytest.fixture
def basedir(drpmocker, tmp_path, monkeypatch):
    drpmocker.add_drp("TEST1", DRP_TEST1)
    monkeypatch.setattr(f"{__name__}.SEEN", [])
    datadir = tmp_path / "data"
    datadir.mkdir()
    # image3.fits is only valid for the old profile of TEST1
    for name, date in [("image1.fits", "2017"), ("image2.fits", "2017"), ("image3.fits", "2015")]:
        hdr = fits.Header()
        hdr["INSTRUME"] = "TEST1"
        hdr["DATE-OBS"] = f"{date}-01-01T00:00:00"
        fits.PrimaryHDU(header=hdr).writeto(datadir / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(basedir, obs, *options, run_values=None, db_values=None, extra=None):
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
    mode_run_common_obs(args, process_unknown_arguments(extra or []), config)
    return SEEN


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
    assert result["qc"] == "GOOD"
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


# Reading the registry

IMAGE = {"id": 10, "mode": "image", "instrument": "TEST1", "images": ["image1.fits"]}


def bias_uuids(basedir):
    registry = Registry(str(basedir / "numina-db.json"))
    return [prod["uuid"] for prod in registry.products(type="MasterBias")]


def test_product_from_registry(basedir):
    """A product reduced in a previous run is used"""
    run(basedir, [BIAS], "--db", "numina-db.json")
    seen = run(basedir, [IMAGE], "--db", "numina-db.json")
    assert seen == bias_uuids(basedir)


def test_most_recent_product(basedir):
    run(basedir, [BIAS], "--db", "numina-db.json")
    run(basedir, [BIAS], "--db", "numina-db.json")
    seen = run(basedir, [IMAGE], "--db", "numina-db.json")
    assert seen == bias_uuids(basedir)[-1:]


def test_bad_product_not_used(basedir):
    run(basedir, [BIAS], "--db", "numina-db.json")
    bad = dict(BIAS, requirements={"quality": "BAD"})
    run(basedir, [bad], "--db", "numina-db.json")
    seen = run(basedir, [IMAGE], "--db", "numina-db.json")
    first, second = bias_uuids(basedir)
    assert seen == [first]


def test_product_of_other_profile_not_used(basedir):
    """A product of another instrument profile is not used"""
    old = dict(BIAS, images=["image3.fits"])
    run(basedir, [old], "--db", "numina-db.json")
    # no master bias for the profile of image1.fits, not in calibsdir either
    with pytest.raises(ValueError, match="Required 'master_bias' of type MasterBias"):
        run(basedir, [IMAGE], "--db", "numina-db.json")
    assert SEEN == []


def test_control_file_without_product_in_registry(basedir):
    """Without products in the registry, the products of the control file are used"""
    hdr = fits.Header()
    hdr["UUID"] = "11111111-1111-1111-1111-111111111111"
    fits.PrimaryHDU(header=hdr).writeto(basedir / "bias_control.fits")
    control = basedir / "control.yaml"
    products = {
        "TEST1": {
            "225fcaf2-7f6f-49cc-972a-70fd0aee8e96": [
                {"id": 1, "type": "MasterBias", "tags": {}, "content": str(basedir / "bias_control.fits")}
            ]
        }
    }
    control.write_text(yaml.safe_dump({"version": 1, "products": products}))
    seen = run(basedir, [IMAGE], "--db", "numina-db.json", "-r", str(control))
    assert seen == [hdr["UUID"]]
    # with a product in the registry, the registry is used
    run(basedir, [BIAS], "--db", "numina-db.json", "-r", str(control))
    seen = run(basedir, [IMAGE], "--db", "numina-db.json", "-r", str(control))
    assert seen[-1] == bias_uuids(basedir)[-1]


def test_ob_over_registry(basedir):
    """The requirements of the OB have priority over the registry"""
    run(basedir, [BIAS], "--db", "numina-db.json")
    run(basedir, [BIAS], "--db", "numina-db.json")
    first = Registry(str(basedir / "numina-db.json")).products(type="MasterBias")[0]
    image = dict(IMAGE, requirements={"master_bias": str(basedir / first["content"])})
    seen = run(basedir, [image], "--db", "numina-db.json")
    assert seen == [first["uuid"]]


def test_cli_over_ob(basedir):
    """The values given in the command line have priority over the OB"""
    param = {"id": 20, "mode": "param", "instrument": "TEST1", "images": ["image1.fits"]}
    assert run(basedir, [param]) == [1]
    SEEN.clear()
    assert run(basedir, [dict(param, requirements={"value": 2})]) == [2]
    SEEN.clear()
    assert run(basedir, [dict(param, requirements={"value": 2})], extra=["--parameter-value=3"]) == [3]


def test_cli_over_registry(basedir):
    run(basedir, [BIAS], "--db", "numina-db.json")
    run(basedir, [BIAS], "--db", "numina-db.json")
    first = Registry(str(basedir / "numina-db.json")).products(type="MasterBias")[0]
    extra = [f"--parameter-master_bias={basedir / first['content']}"]
    seen = run(basedir, [IMAGE], "--db", "numina-db.json", extra=extra)
    assert seen == [first["uuid"]]
