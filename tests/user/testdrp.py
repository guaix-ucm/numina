"""DRPs, recipes and observing blocks shared by the tests of numina run

The recipes record the values they receive in SEEN (a list) or in
SEEN_BY_MODE (a dict of lists). conftest.py empties both before each
test; the tests read them with the fixtures seen and seen_by_mode.
"""

import argparse
import pathlib
import uuid

import astropy.io.fits as fits
import yaml

from numina.core import BaseRecipe, DataFrameType, Requirement, Result
from numina.core.query import ResultOf
from numina.core.requirements import ObservationResultRequirement
from numina.testing.recipes import MasterBias
from numina.types.qc import QC
import numina.core.dataholders as dh
from numina.user.cli import base_config, main, process_unknown_arguments
from numina.user.clirun import register
from numina.user.clirundal import mode_run_common_obs

# values received by the recipes
SEEN = []
# values received by the recipes, by mode
SEEN_BY_MODE = {}


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
    """Records the UUID of the master bias it receives"""

    obresult = ObservationResultRequirement()
    master_bias = Requirement(MasterBias, "Master bias")

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        with recipe_input.master_bias.open() as hdul:
            SEEN.append(hdul[0].header["UUID"])
        return self.create_result()


class ParamRecipe(BaseRecipe):
    """Records the value of its parameter"""

    value = dh.Parameter(1, "a value")

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        SEEN.append(recipe_input.value)
        return self.create_result()


class UseLastRecipe(BaseRecipe):
    """Records the UUID of the last master bias"""

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


class ChildRecipe(BaseRecipe):
    """Records its profile, in SEEN_BY_MODE['child']"""

    obresult = ObservationResultRequirement()
    reduced_image = dh.Result(DataFrameType)

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        obresult = recipe_input.obresult
        SEEN_BY_MODE.setdefault("child", []).append(obresult.profile)
        with obresult.frames[0].open() as hdul:
            hdr = hdul[0].header.copy()
        hdr["UUID"] = str(uuid.uuid4())
        reduced = fits.HDUList([fits.PrimaryHDU(header=hdr)])
        return self.create_result(reduced_image=reduced)


class ParentRecipe(BaseRecipe):
    """Records its profile and configuration, in SEEN_BY_MODE"""

    obresult = ObservationResultRequirement(query_opts=ResultOf("reduced_image", node="children"))

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        obresult = recipe_input.obresult
        SEEN_BY_MODE.setdefault("parent", []).append(obresult.profile)
        SEEN_BY_MODE.setdefault("parent_conf", []).append(str(obresult.configuration.origin.uuid))
        return self.create_result()


# DRP with a bias, an image that uses the bias, a recipe that fails
# and a recipe with a parameter
DRP_TEST1 = """
name: TEST1
configurations:
  path: numina.testing.drps.configs
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
      bias: tests.user.testdrp.BiasRecipe
      fail: numina.core.utils.AlwaysFailRecipe
      image: tests.user.testdrp.ImageRecipe
      param: tests.user.testdrp.ParamRecipe
"""

# DRP with OBs that use the results of their children
DRP_NESTED = """
name: TEST1
configurations:
  path: numina.testing.drps.configs
modes:
  - key: child
    name: Child
    summary: Child mode
    description: Child mode
  - key: parent
    name: Parent
    summary: Parent mode
    description: Parent mode
pipelines:
  default:
    version: 1
    recipes:
      child: tests.user.testdrp.ChildRecipe
      parent: tests.user.testdrp.ParentRecipe
"""

BIAS = {"id": 1, "mode": "bias", "instrument": "TEST1", "images": ["image1.fits"]}
IMAGE = {"id": 10, "mode": "image", "instrument": "TEST1", "images": ["image1.fits"]}


def write_images(datadir, images):
    """Write empty images of TEST1, images is a list of (name, DATE-OBS)"""
    datadir.mkdir(exist_ok=True)
    for name, date in images:
        hdr = fits.Header()
        hdr["INSTRUME"] = "TEST1"
        hdr["DATE-OBS"] = date
        fits.PrimaryHDU(header=hdr).writeto(datadir / name)


def create_basedir(drpmocker, tmp_path, monkeypatch, drp=DRP_TEST1):
    """The DRP TEST1 and its images, in tmp_path, the working directory"""
    drpmocker.add_drp("TEST1", drp)
    # image3.fits is only valid for the old profile of TEST1
    write_images(
        tmp_path / "data",
        [
            ("image1.fits", "2017-01-01T00:00:00"),
            ("image2.fits", "2017-01-01T00:00:00"),
            ("image3.fits", "2015-01-01T00:00:00"),
        ],
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


# options of numina run to record the reductions in a registry
DB_OPTIONS = ("--db", "numina-db.json")


def run_parser():
    """The parser of the command line, with the command run"""
    parser = argparse.ArgumentParser(prog="numina")
    register(parser.add_subparsers(), base_config())
    return parser


def make_config(basedir, run_values=None, db_values=None):
    """The default configuration, with basedir and other values of [tool.run] and [tool.db]"""
    config = base_config()
    config["tool.run"]["basedir"] = str(basedir)
    for key, value in (run_values or {}).items():
        config["tool.run"][key] = value
    for key, value in (db_values or {}).items():
        config["tool.db"][key] = value
    return config


def run_obs(config, obs, *options, extra=None):
    """numina run with the observing blocks obs and the options

    obs is an observing block or a list of them, written in obsdata.yaml
    in the basedir of config. extra are the arguments not known by the
    parser, such as --parameter-NAME=VALUE.
    """
    if isinstance(obs, dict):
        obs = [obs]
    obsfile = pathlib.Path(config["tool.run"]["basedir"]) / "obsdata.yaml"
    obsfile.write_text(yaml.safe_dump_all(obs))
    args = run_parser().parse_args(["run", *options, str(obsfile)])
    mode_run_common_obs(args, process_unknown_arguments(extra or []), config)


def cli(capsys, *args):
    """numina with the arguments, without plugins

    Returns the exit status, the lines of the standard output and the
    standard error.
    """
    status = main(["--disable-plugins", *args])
    captured = capsys.readouterr()
    return status, captured.out.splitlines(), captured.err
