"""Options --copy-files and --link-files of numina run"""

import pytest

from .testdrp import run_obs, run_parser, write_images

DRP_TEST1 = """
name: TEST1
configurations:
  path: tests.support.drps.configs
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
def image_drp(drpmocker, run_config, tmp_path):
    drpmocker.add_drp("TEST1", DRP_TEST1)
    write_images(tmp_path / "data", [("image1.fits", "2017-01-01T00:00:00")])
    run_config["tool.run"]["datadir"] = str(tmp_path / "data")


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
def test_copy_or_link(run_config, image_drp, tmp_path, options, config_copy, linked):
    run_config["tool.run"]["copy_files"] = config_copy

    run_obs(run_config, {"id": 1, "mode": "image", "instrument": "TEST1", "images": ["image1.fits"]}, *options)

    installed = tmp_path / "obsid1_work" / "image1.fits"
    assert installed.is_file()
    assert installed.is_symlink() == linked


def test_copy_and_link_are_exclusive(capsys):
    with pytest.raises(SystemExit):
        run_parser().parse_args(["run", "--copy-files", "--link-files", "obsdata.yaml"])
    assert "not allowed with argument" in capsys.readouterr().err
