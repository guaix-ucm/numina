"""The instrument profile of an OB with nested results (issue #233)"""

import argparse
import json
import uuid

import astropy.io.fits as fits
import pytest
import yaml

from numina.core import BaseRecipe, DataFrameType
from numina.core.query import ResultOf
from numina.core.requirements import ObservationResultRequirement
import numina.core.dataholders as dh

from ..cli import base_config, process_unknown_arguments
from ..clirun import register
from ..clirundal import mode_run_common_obs

DRP_TEST1 = """
name: TEST1
configurations:
  path: numina.drps.tests.configs
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
      child: numina.user.tests.test_nested_profile.ChildRecipe
      parent: numina.user.tests.test_nested_profile.ParentRecipe
"""

TEST1_PROFILE = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"  # from 2016-06-01
TEST1_PROFILE_OLD = "6ad5dc90-6b15-43b7-abb5-b07340e19f41"  # 2014-06-01 to 2016-06-01

# profiles received by the recipes, by mode
SEEN = {}


class ChildRecipe(BaseRecipe):
    obresult = ObservationResultRequirement()
    reduced_image = dh.Result(DataFrameType)

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        obresult = recipe_input.obresult
        SEEN.setdefault("child", []).append(obresult.profile)
        with obresult.frames[0].open() as hdul:
            hdr = hdul[0].header.copy()
        hdr["UUID"] = str(uuid.uuid4())
        reduced = fits.HDUList([fits.PrimaryHDU(header=hdr)])
        return self.create_result(reduced_image=reduced)


class ParentRecipe(BaseRecipe):
    obresult = ObservationResultRequirement(query_opts=ResultOf("reduced_image", node="children"))

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        obresult = recipe_input.obresult
        SEEN.setdefault("parent", []).append(obresult.profile)
        SEEN.setdefault("parent_conf", []).append(str(obresult.configuration.origin.uuid))
        return self.create_result()


@pytest.fixture
def run_nested(drpmocker, run_config, tmp_path, monkeypatch):
    monkeypatch.setattr(f"{__name__}.SEEN", {})
    drpmocker.add_drp("TEST1", DRP_TEST1)
    datadir = tmp_path / "data"
    datadir.mkdir()
    for name in ["image1.fits", "image2.fits"]:
        hdr = fits.Header()
        hdr["INSTRUME"] = "TEST1"
        # valid only for the old profile
        hdr["DATE-OBS"] = "2015-01-01T00:00:00"
        fits.PrimaryHDU(header=hdr).writeto(datadir / name)
    run_config["tool.run"]["datadir"] = str(datadir)
    obs = [
        {"id": 1, "mode": "child", "instrument": "TEST1", "images": ["image1.fits"]},
        {"id": 2, "mode": "child", "instrument": "TEST1", "images": ["image2.fits"]},
        {"id": 3, "mode": "parent", "instrument": "TEST1", "children": [1, 2]},
    ]
    obsfile = tmp_path / "obsdata.yaml"
    obsfile.write_text(yaml.safe_dump_all(obs))

    def run(*options):
        parser = argparse.ArgumentParser(prog="numina")
        register(parser.add_subparsers(), base_config())
        args = parser.parse_args(["run", *options, str(obsfile)])
        mode_run_common_obs(args, process_unknown_arguments([]), run_config)
        return SEEN

    return run


def stored_profiles(tmp_path):
    """Profiles stored in task.json, by OB id"""
    profiles = {}
    for obsid in [1, 2, 3]:
        with open(tmp_path / f"obsid{obsid}_results" / "task.json") as fd:
            profiles[obsid] = json.load(fd)["request_params"]["instrument_configuration"]
    return profiles


def test_nested_profile_from_children(run_nested, tmp_path):
    seen = run_nested()

    assert seen["child"] == [TEST1_PROFILE_OLD, TEST1_PROFILE_OLD]
    # the profile of the parent is selected from the results of the children
    assert seen["parent"] == [TEST1_PROFILE_OLD]
    assert seen["parent_conf"] == [TEST1_PROFILE_OLD]
    assert stored_profiles(tmp_path) == {1: TEST1_PROFILE_OLD, 2: TEST1_PROFILE_OLD, 3: TEST1_PROFILE_OLD}


def test_nested_profile_forced(run_nested, tmp_path):
    seen = run_nested("--insconf", TEST1_PROFILE)

    assert seen["child"] == [TEST1_PROFILE, TEST1_PROFILE]
    assert seen["parent"] == [TEST1_PROFILE]
    assert seen["parent_conf"] == [TEST1_PROFILE]
    assert stored_profiles(tmp_path) == {1: TEST1_PROFILE, 2: TEST1_PROFILE, 3: TEST1_PROFILE}
