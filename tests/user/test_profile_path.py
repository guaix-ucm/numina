"""Additional instrument configurations with --profile-path"""

import argparse
import importlib.resources
import json
import logging

import pytest
import yaml

from numina.instrument.collection import load_paths_store
from numina.user.baserun import run_reduce
from numina.user.cli import base_config, process_unknown_arguments
from numina.user.clirun import register
from numina.user.clirundal import mode_run_common_obs
from numina.user.helpers import create_datamanager

# A configuration of TEST1 that is not in numina.testing.drps.configs
NEW_PROFILE = "0c6a1e5e-8d4f-4d8b-9b0e-2d7f0e3c5a11"

DRP_TEST1 = """
name: TEST1
configurations:
  path: numina.testing.drps.configs
modes:
  - key: image
    name: Image
    summary: Image mode
    description: Image mode
pipelines:
  default:
    version: 1
    recipes:
      image: numina.core.utils.AlwaysSuccessRecipe
"""


def make_test1_configuration(uuid, date_start):
    """A copy of the configuration of TEST1, with other uuid and start date"""
    base = importlib.resources.files("numina.testing.drps.configs").joinpath("instrument-test1.json")
    conf = json.loads(base.read_text())
    conf["uuid"] = uuid
    conf["date_start"] = date_start
    return conf


@pytest.fixture
def extra_dir(tmp_path):
    """A directory with a new configuration of TEST1"""
    extra = tmp_path / "extra"
    extra.mkdir()
    conf = make_test1_configuration(NEW_PROFILE, "2020-01-01T00:00:00")
    (extra / "instrument-test1-new.json").write_text(json.dumps(conf))
    return extra


@pytest.fixture
def run_setup(drpmocker, run_config, tmp_path):
    drpmocker.add_drp("TEST1", DRP_TEST1)
    datadir = tmp_path / "data"
    datadir.mkdir()
    run_config["tool.run"]["datadir"] = str(datadir)
    return run_config


def add_ob(datamanager):
    datamanager.backend.add_obs([{"id": 1, "mode": "image", "instrument": "TEST1", "images": []}])


def test_profile_path_adds_configuration(run_setup, extra_dir):
    datamanager = create_datamanager(run_setup, None, profile_path_extra=str(extra_dir))
    add_ob(datamanager)

    task = run_reduce(datamanager, 1, profile=NEW_PROFILE)

    assert task.request_params["instrument_configuration"] == NEW_PROFILE


def test_without_profile_path(run_setup):
    datamanager = create_datamanager(run_setup, None)
    add_ob(datamanager)

    with pytest.raises(ValueError, match=f"Not found instrument uuid={NEW_PROFILE}"):
        run_reduce(datamanager, 1, profile=NEW_PROFILE)


def test_numina_run_profile_path(run_setup, extra_dir, tmp_path):
    """The option --profile-path of numina run reaches create_datamanager"""
    obsfile = tmp_path / "obsdata.yaml"
    obsfile.write_text(yaml.safe_dump({"id": 1, "mode": "image", "instrument": "TEST1", "images": []}))
    parser = argparse.ArgumentParser(prog="numina")
    register(parser.add_subparsers(), base_config())
    args = parser.parse_args(["run", "--profile-path", str(extra_dir), "--insconf", NEW_PROFILE, str(obsfile)])

    mode_run_common_obs(args, process_unknown_arguments([]), run_setup)

    task = json.loads((tmp_path / "obsid1_results" / "task.json").read_text())
    assert task["request_params"]["instrument_configuration"] == NEW_PROFILE


def test_replaced_file_warning(tmp_path, caplog):
    """A file with the same name as one of the DRP replaces it, with a warning"""
    extra = tmp_path / "extra"
    extra.mkdir()
    conf = make_test1_configuration(NEW_PROFILE, "2020-01-01T00:00:00")
    (extra / "instrument-test1.json").write_text(json.dumps(conf))

    with caplog.at_level(logging.WARNING, logger="numina.instrument.collection"):
        store = load_paths_store(["numina.testing.drps.configs"], [str(extra)])

    assert "configuration file instrument-test1.json in" in caplog.text
    assert "replaces the one in" in caplog.text
    # The file of the DRP is used
    assert str(store["instrument-test1.json"]["origin"].uuid) != NEW_PROFILE
