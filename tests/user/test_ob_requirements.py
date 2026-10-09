"""The requirements of the observation result reach the recipe"""

import argparse

import yaml

from numina.core import BaseRecipe
import numina.core.dataholders as dh

from numina.user.cli import base_config, process_unknown_arguments
from numina.user.clirun import register
from numina.user.clirundal import mode_run_common_obs

DRP_TEST1 = """
name: TEST1
configurations:
  path: numina.drps.tests.configs
modes:
  - key: param
    name: Param
    summary: Param mode
    description: Param mode
pipelines:
  default:
    version: 1
    recipes:
      param: tests.user.test_ob_requirements.ParamRecipe
"""

# values received by the recipe
SEEN = []


class ParamRecipe(BaseRecipe):
    value = dh.Parameter(1, "a value")

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        SEEN.append(recipe_input.value)
        return self.create_result()


def test_ob_requirements(drpmocker, run_config, tmp_path, monkeypatch):
    monkeypatch.setattr(f"{__name__}.SEEN", [])
    drpmocker.add_drp("TEST1", DRP_TEST1)
    datadir = tmp_path / "data"
    datadir.mkdir()
    run_config["tool.run"]["datadir"] = str(datadir)
    obs = [
        {"id": 1, "mode": "param", "instrument": "TEST1", "images": []},
        {"id": 2, "mode": "param", "instrument": "TEST1", "images": [], "requirements": {"value": 2}},
    ]
    obsfile = tmp_path / "obsdata.yaml"
    obsfile.write_text(yaml.safe_dump_all(obs))

    parser = argparse.ArgumentParser(prog="numina")
    register(parser.add_subparsers(), base_config())
    args = parser.parse_args(["run", str(obsfile)])
    mode_run_common_obs(args, process_unknown_arguments([]), run_config)

    assert SEEN == [1, 2]
