"""Selection of the instrument configuration (profile) in numina run"""

import argparse

import astropy.io.fits as fits
import pytest

from numina.user.baserun import run_reduce
from numina.user.clirun import profile_uuid
from numina.user.helpers import create_datamanager

# Profiles in numina/drps/tests/configs
TEST1_PROFILE = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"
TEST1_OLD_PROFILE = "6ad5dc90-6b15-43b7-abb5-b07340e19f41"
TEST2_PROFILE = "9c21b315-9231-4fe0-a276-5043b064a3a8"
UNKNOWN_PROFILE = "11111111-2222-3333-4444-555555555555"

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


def test_profile_uuid():
    assert profile_uuid(TEST1_PROFILE.upper()) == TEST1_PROFILE


def test_profile_uuid_invalid():
    with pytest.raises(argparse.ArgumentTypeError, match="'TEST1' is not a valid UUID"):
        profile_uuid("TEST1")


@pytest.fixture
def datamanager(drpmocker, run_config, tmp_path):
    drpmocker.add_drp("TEST1", DRP_TEST1)
    datadir = tmp_path / "data"
    datadir.mkdir()
    # The automatic selection uses the date of the image,
    # TEST1_PROFILE is valid since 2016
    hdr = fits.Header()
    hdr["INSTRUME"] = "TEST1"
    hdr["DATE-OBS"] = "2017-01-01T00:00:00"
    fits.PrimaryHDU(header=hdr).writeto(datadir / "image1.fits")
    run_config["tool.run"]["datadir"] = str(datadir)
    datamanager = create_datamanager(run_config, None)
    datamanager.backend.add_obs([{"id": 1, "mode": "image", "instrument": "TEST1", "images": ["image1.fits"]}])
    return datamanager


@pytest.mark.parametrize(
    "profile, expected",
    [
        (None, TEST1_PROFILE),  # selected from the date of the image
        (TEST1_OLD_PROFILE, TEST1_OLD_PROFILE),  # forced, the date of the image is not used
    ],
)
def test_profile_selection(datamanager, profile, expected):
    task = run_reduce(datamanager, 1, profile=profile)

    assert task.state == 2
    # The profile used is recorded in the task
    assert task.request_params["instrument_configuration"] == expected


def test_profile_not_found(datamanager):
    with pytest.raises(ValueError, match=f"Not found instrument uuid={UNKNOWN_PROFILE}"):
        run_reduce(datamanager, 1, profile=UNKNOWN_PROFILE)


def test_profile_of_other_instrument(datamanager):
    with pytest.raises(ValueError, match="is for instrument 'TEST2', not 'TEST1'"):
        run_reduce(datamanager, 1, profile=TEST2_PROFILE)


def test_profile_without_images(datamanager, caplog):
    """Without images, the most recent configuration of the instrument is used"""
    datamanager.backend.add_obs([{"id": 2, "mode": "image", "instrument": "TEST1", "images": []}])

    task = run_reduce(datamanager, 2)

    assert task.request_params["instrument_configuration"] == TEST1_PROFILE
    assert "using the most recent" in caplog.text


@pytest.fixture
def run_parser():
    from numina.user.cli import base_config
    from numina.user.clirun import register

    parser = argparse.ArgumentParser(prog="numina")
    register(parser.add_subparsers(), base_config())
    return parser


@pytest.mark.parametrize("option", ["--insconf", "--profile"])
def test_profile_option_names(run_parser, option):
    args = run_parser.parse_args(["run", option, TEST1_OLD_PROFILE, "obsdata.yaml"])
    assert args.profile == TEST1_OLD_PROFILE


def test_profile_option_ambiguous(run_parser, capsys):
    """--prof matches --profile and --profile-path"""
    with pytest.raises(SystemExit):
        run_parser.parse_args(["run", "--prof", TEST1_OLD_PROFILE, "obsdata.yaml"])
    assert "ambiguous option" in capsys.readouterr().err
