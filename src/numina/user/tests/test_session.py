"""Reduction sessions (numina.user.session)"""

import astropy.io.fits as fits
import pytest
import yaml

from numina.user.cli import base_config
from numina.user.session import Session

from . import test_run_registry as trr

BIAS = trr.BIAS
IMAGE = trr.IMAGE


@pytest.fixture
def basedir(drpmocker, tmp_path, monkeypatch):
    drpmocker.add_drp("TEST1", trr.DRP_TEST1)
    monkeypatch.setattr(trr, "SEEN", [])
    datadir = tmp_path / "data"
    datadir.mkdir()
    hdr = fits.Header()
    hdr["INSTRUME"] = "TEST1"
    hdr["DATE-OBS"] = "2017-01-01T00:00:00"
    fits.PrimaryHDU(header=hdr).writeto(datadir / "image1.fits")
    # obs files, relative to basedir
    (tmp_path / "bias.yaml").write_text(yaml.safe_dump(BIAS))
    (tmp_path / "image.yaml").write_text(yaml.safe_dump(IMAGE))
    return tmp_path


def new_session(basedir, **kwargs):
    # base_config, so that the configuration files of the user are not read
    return Session(basedir=basedir, config=base_config(), **kwargs)


def test_session_with_registry(basedir):
    session = new_session(basedir, db="numina-db.json")
    session.add_observations("bias.yaml")
    task = session.run(1)

    assert task.state == 2
    assert task.result.master_bias is not None
    assert (basedir / "numina-db.json").is_file()
    (product,) = session.products(type="MasterBias")
    assert product["task_id"] == task.id
    assert [t["oblock_id"] for t in session.tasks()] == [1]
    assert len(session.results(oblock_id=1)) == 1


def test_products_of_previous_session(basedir):
    """A new session uses the products recorded by a previous one"""
    first = new_session(basedir, db="numina-db.json")
    first.add_observations("bias.yaml")
    first.run(1)

    second = new_session(basedir, db="numina-db.json")
    second.add_observations("image.yaml")
    second.run(10)

    (product,) = second.products(type="MasterBias")
    assert trr.SEEN == [product["uuid"]]


def test_observations_as_dicts(basedir):
    session = new_session(basedir, db="numina-db.json")
    session.add_observations(BIAS, IMAGE)
    session.run(1)
    session.run(10)
    assert [t["oblock_id"] for t in session.tasks()] == [1, 10]


def test_session_with_control(basedir):
    """The control file is relative to basedir"""
    hdr = fits.Header()
    hdr["UUID"] = "33333333-3333-3333-3333-333333333333"
    fits.PrimaryHDU(header=hdr).writeto(basedir / "data" / "bias_control.fits")
    products = {
        "TEST1": {
            "225fcaf2-7f6f-49cc-972a-70fd0aee8e96": [
                {"id": 1, "type": "MasterBias", "tags": {}, "content": "bias_control.fits"}
            ]
        }
    }
    (basedir / "control.yaml").write_text(yaml.safe_dump({"version": 1, "products": products}))
    session = new_session(basedir, control="control.yaml")
    session.add_observations("image.yaml")
    session.run(10)
    assert trr.SEEN == [hdr["UUID"]]


def test_session_without_registry(basedir):
    session = new_session(basedir)
    session.add_observations("bias.yaml")
    task = session.run(1)

    assert task.state == 2
    assert session.registry is None
    assert not list(basedir.glob("*.json"))
    with pytest.raises(ValueError, match="no registry of reductions"):
        session.products()


def test_copy_files_from_config(basedir):
    config = base_config()
    config["tool.run"]["copy_files"] = "True"
    session = Session(basedir=basedir, config=config)
    session.add_observations("bias.yaml")
    session.run(1)
    installed = basedir / "obsid1_work" / "image1.fits"
    assert installed.is_file() and not installed.is_symlink()
