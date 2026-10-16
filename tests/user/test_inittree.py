"""numina inittree: installation of the datasets of calibrations of the DRPs"""

import configparser
import hashlib
import zipfile

import pytest

import numina.core.pipelineload as pload
from numina.util.download import check_hash, file_hash

from .testdrp import cli

DRP_TEST1 = """
name: TEST1
configurations:
  path: numina.testing.drps.configs
modes: []
pipelines:
  default:
    version: 1
    recipes: {}
"""

FILES = {
    "calibrations-1.0/README.md": b"readme",
    "calibrations-1.0/.gitattributes": b"*.fits binary",
    "calibrations-1.0/P1/MasterBias/master_bias.fits": b"bias",
    "calibrations-1.0/P1/TraceMap/LCB/HR-R/master_traces.json": b"{}",
}


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path


@pytest.fixture
def setup(drpmocker, tmp_path, monkeypatch):
    """A DRP with two datasets, the work directory and an empty cache"""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    archive = make_zip(tmp_path / "calibrations-1.0.zip", FILES)
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    datasets = [
        {"name": "old", "url": "file:///nonexistent.zip", "known_hash": "md5:0"},
        {"name": "calibrations-1.0", "url": archive.as_uri(), "known_hash": f"md5:{md5}", "description": "test"},
    ]

    def load_drp():
        drp = pload.drp_load_data("numina", DRP_TEST1)
        drp.calibrations = datasets
        return drp

    drpmocker.add_drp("TEST1", load_drp)
    return work, datasets


def inittree(capsys, *args):
    return cli(capsys, "inittree", *args)


def test_list(setup, capsys):
    status, lines, _ = inittree(capsys)
    assert status == 0
    assert lines == ["TEST1  old", "TEST1  calibrations-1.0  test"]


def test_install(setup, capsys):
    work, _ = setup
    status, lines, _ = inittree(capsys, "TEST1")
    assert status == 0
    assert "3 new files, 0 unchanged, 0 with other contents, in calibrations/TEST1" in lines
    # the root directory of the ZIP file is removed, the hidden files are skipped
    assert (work / "calibrations/TEST1/P1/MasterBias/master_bias.fits").read_bytes() == b"bias"
    assert (work / "calibrations/TEST1/README.md").is_file()
    assert not (work / "calibrations/TEST1/.gitattributes").exists()
    config = configparser.ConfigParser()
    config.read(work / ".numina.cfg")
    assert config["tool.run"]["calibsdir"] == "calibrations"
    assert config["tool.inittree.TEST1"]["dataset"] == "calibrations-1.0"

    # again, nothing changes
    status, lines, _ = inittree(capsys, "TEST1")
    assert "0 new files, 3 unchanged, 0 with other contents, in calibrations/TEST1" in lines


def test_dry_run(setup, capsys):
    work, _ = setup
    status, lines, _ = inittree(capsys, "TEST1", "--dry-run")
    assert status == 0
    assert lines[-1] == "dry run, nothing was changed"
    assert not (work / "calibrations").exists()
    assert not (work / ".numina.cfg").exists()


def test_conflicts(setup, capsys):
    work, _ = setup
    inittree(capsys, "TEST1", "--no-config")
    readme = work / "calibrations/TEST1/README.md"
    readme.write_bytes(b"changed")
    (work / "calibrations/TEST1/P1/MasterBias/master_bias.fits").unlink()

    status, lines, err = inittree(capsys, "TEST1", "--no-config")
    assert status == 2
    assert "nothing was installed, use --overwrite" in err
    assert readme.read_bytes() == b"changed"
    # the new files are not installed either
    assert not (work / "calibrations/TEST1/P1/MasterBias/master_bias.fits").exists()

    status, lines, _ = inittree(capsys, "TEST1", "--no-config", "--overwrite")
    assert status == 0
    assert readme.read_bytes() == b"readme"
    assert (work / "calibrations/TEST1/P1/MasterBias/master_bias.fits").is_file()
    assert not (work / ".numina.cfg").exists()


def test_base_path_and_directory(setup, capsys):
    work, datasets = setup
    datasets[1]["directory"] = "OTHER"
    status, _, _ = inittree(capsys, "TEST1", "--base-path", "data/cal")
    assert status == 0
    assert (work / "data/cal/OTHER/README.md").is_file()
    config = configparser.ConfigParser()
    config.read(work / ".numina.cfg")
    assert config["tool.run"]["calibsdir"] == "data/cal"


def test_wrong_hash(setup, capsys, tmp_path):
    work, datasets = setup
    datasets[1]["known_hash"] = "md5:0123"
    status, _, err = inittree(capsys, "TEST1")
    assert status == 2
    assert "the md5 hash of the file is" in err
    assert not (work / "calibrations").exists()
    # the file is removed from the cache
    from astropy.utils import data

    assert not data.is_url_in_cache(datasets[1]["url"], pkgname="numina")


@pytest.mark.parametrize(
    "args, match",
    [
        (["TEST1", "--dataset", "nope"], "has no dataset nope, its datasets are old, calibrations-1.0"),
        (["OTHER"], "no DRP for instrument OTHER"),
    ],
)
def test_errors(setup, capsys, args, match):
    status, _, err = inittree(capsys, *args)
    assert status == 2
    assert match in err


def test_no_datasets(drpmocker, capsys):
    drpmocker.add_drp("TEST1", lambda: pload.drp_load_data("numina", DRP_TEST1))
    status, _, err = inittree(capsys, "TEST1")
    assert status == 2
    assert "has no datasets of calibrations" in err


def test_several_roots_and_unsafe_members(tmp_path):
    from numina.user.inittree import plan_install

    archive = make_zip(tmp_path / "a.zip", {"A/x.txt": b"x", "B/y.txt": b"y", "../evil.txt": b"e"})
    plan = plan_install(archive, tmp_path / "dest")
    assert sorted(str(target.relative_to(tmp_path / "dest")) for _, target in plan.new) == ["A/x.txt", "B/y.txt"]
    assert [name for name, _ in plan.skipped] == ["../evil.txt"]


def test_check_hash(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello")
    check_hash(path, "md5:5d41402abc4b2a76b9719d911017c592")
    check_hash(path, f"sha256:{file_hash(path)}")
    with pytest.raises(ValueError, match="the md5 hash of the file is"):
        check_hash(path, "md5:0")
    with pytest.raises(ValueError, match="invalid hash"):
        check_hash(path, "crc:0")
