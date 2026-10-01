"""Options --copy-files and --link-files of numina run"""

import argparse

import astropy.io.fits as fits
import pytest
import yaml

from ..cli import base_config, process_unknown_arguments
from ..clirun import register
from ..clirundal import mode_run_common_obs

DRP_TEST1 = """
name: TEST1
configurations:
  path: numina.drps.tests.configs
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


@pytest.fixture
def run_parser():
    parser = argparse.ArgumentParser(prog="numina")
    register(parser.add_subparsers(), base_config())
    return parser


@pytest.fixture
def obsfile(drpmocker, run_config, tmp_path):
    drpmocker.add_drp("TEST1", DRP_TEST1)
    datadir = tmp_path / "data"
    datadir.mkdir()
    hdr = fits.Header()
    hdr["INSTRUME"] = "TEST1"
    hdr["DATE-OBS"] = "2017-01-01T00:00:00"
    fits.PrimaryHDU(header=hdr).writeto(datadir / "image1.fits")
    run_config["tool.run"]["datadir"] = str(datadir)
    obsfile = tmp_path / "obsdata.yaml"
    obsfile.write_text(yaml.safe_dump({"id": 1, "mode": "image", "instrument": "TEST1", "images": ["image1.fits"]}))
    return obsfile


@pytest.mark.parametrize(
    "options, config_copy, linked",
    [
        ([], "False", True),  # the default of numina.cfg
        ([], "True", False),  # from the configuration
        (["--copy-files"], "False", False),
        (["--link-files"], "True", True),  # the option overrides the configuration
        (["--not-copy-files"], "True", True),
    ],
)
def test_copy_or_link(run_parser, run_config, obsfile, tmp_path, options, config_copy, linked):
    run_config["tool.run"]["copy_files"] = config_copy
    args = run_parser.parse_args(["run", *options, str(obsfile)])

    mode_run_common_obs(args, process_unknown_arguments([]), run_config)

    installed = tmp_path / "obsid1_work" / "image1.fits"
    assert installed.is_file()
    assert installed.is_symlink() == linked


def test_copy_and_link_are_exclusive(run_parser, capsys):
    with pytest.raises(SystemExit):
        run_parser.parse_args(["run", "--copy-files", "--link-files", "obsdata.yaml"])
    assert "not allowed with argument" in capsys.readouterr().err
