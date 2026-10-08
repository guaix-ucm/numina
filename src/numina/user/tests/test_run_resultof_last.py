"""ResultOf with node='last' and a registry of reductions"""

import argparse

import astropy.io.fits as fits
import pytest
import yaml

from numina.core import BaseRecipe, DataFrameType, Requirement
from numina.core.query import ResultOf
from numina.core.requirements import ObservationResultRequirement
from numina.dal.registry import Registry

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
  - key: uselast
    name: Use last
    summary: Use last mode
    description: Use last mode
pipelines:
  default:
    version: 1
    recipes:
      bias: numina.user.tests.test_run_registry.BiasRecipe
      uselast: numina.user.tests.test_run_resultof_last.UseLastRecipe
"""

SEEN = []


class UseLastRecipe(BaseRecipe):
    obresult = ObservationResultRequirement()
    master_bias = Requirement(
        DataFrameType, "The last master bias", query_opts=ResultOf("bias.master_bias", node="last")
    )

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        with recipe_input.master_bias.open() as hdul:
            SEEN.append(hdul[0].header["UUID"])
        return self.create_result()


def bias(obid, image="image1.fits", **requirements):
    ob = {"id": obid, "mode": "bias", "instrument": "TEST1", "images": [image]}
    if requirements:
        ob["requirements"] = requirements
    return ob


USE_LAST = {"id": 100, "mode": "uselast", "instrument": "TEST1", "images": ["image1.fits"]}


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
    return SEEN


def bias_uuids(basedir):
    registry = Registry(str(basedir / "numina-db.json"))
    return [prod["uuid"] for prod in registry.products(type="MasterBias")]


def test_last_result(basedir):
    """The most recent result of the mode, of any observing block"""
    run(basedir, [bias(1), bias(2, image="image2.fits")])
    seen = run(basedir, [USE_LAST])
    assert seen == bias_uuids(basedir)[-1:]


def test_last_result_not_bad(basedir):
    run(basedir, [bias(1), bias(2, image="image2.fits", quality="BAD")])
    seen = run(basedir, [USE_LAST])
    first, second = bias_uuids(basedir)
    assert seen == [first]


def test_last_result_same_profile(basedir):
    """A more recent result of another instrument profile is not used"""
    run(basedir, [bias(1), bias(2, image="image3.fits")])
    seen = run(basedir, [USE_LAST])
    first, other_profile = bias_uuids(basedir)
    assert seen == [first]


def test_no_result(basedir):
    with pytest.raises(ValueError, match="Required 'master_bias'"):
        run(basedir, [USE_LAST])


def test_without_registry(basedir):
    run(basedir, [bias(1)], db=False)
    with pytest.raises(ValueError, match="Required 'master_bias'"):
        run(basedir, [USE_LAST], db=False)
