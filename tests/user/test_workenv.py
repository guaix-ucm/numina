"""Installation of the files of a task in the work directory"""

import os
import shutil

import pytest

import numina.user.helpers as helpers
from numina.core.dataholders import Requirement
from numina.core.oresult import ObservationResult
from numina.core.recipeinout import RecipeInput
from numina.types.dataframe import DataFrame
from numina.types.frame import DataFrameType
from numina.user.helpers import WorkEnvironment, is_copy_of


@pytest.fixture
def datadir(tmp_path):
    data = tmp_path / "data"
    for subdir, content in [("a", "AAAA"), ("b", "BBBB")]:
        (data / subdir).mkdir(parents=True)
        (data / subdir / "master.fits").write_text(content)
    return data


def create_workenv(tmp_path, datadir, workdir="work"):
    work = WorkEnvironment(str(datadir), str(tmp_path), workdir, "results")
    work.sane_work()
    return work


@pytest.fixture
def copies(monkeypatch):
    """Record the calls to shutil.copy2 in helpers"""
    calls = []
    original = shutil.copy2

    def copy2(src, dest):
        calls.append(src)
        return original(src, dest)

    monkeypatch.setattr(helpers.shutil, "copy2", copy2)
    return calls


def test_copy_not_needed(tmp_path, datadir, copies):
    src = str(datadir / "a" / "master.fits")
    for _ in range(2):
        work = create_workenv(tmp_path, datadir)
        work.copy_if_needed("master.fits", src, os.path.join(work.workdir, "master.fits"))

    # copied only in the first run
    assert copies == [src]
    assert not os.path.exists(os.path.join(work.workdir, "index.pkl"))


def test_copy_again_if_source_changes(tmp_path, datadir, copies):
    src = datadir / "a" / "master.fits"
    work = create_workenv(tmp_path, datadir)
    dest = os.path.join(work.workdir, "master.fits")
    work.copy_if_needed("master.fits", str(src), dest)

    # same size, other modification time
    src.write_text("CCCC")
    os.utime(src, (0, os.stat(dest).st_mtime + 10))
    work.copy_if_needed("master.fits", str(src), dest)

    assert len(copies) == 2
    with open(dest) as fd:
        assert fd.read() == "CCCC"


def test_copy_after_link(tmp_path, datadir):
    """A link of a previous run is replaced by a copy"""
    src = str(datadir / "a" / "master.fits")
    work = create_workenv(tmp_path, datadir)
    dest = os.path.join(work.workdir, "master.fits")
    work.link_if_needed("master.fits", src, dest)
    assert os.path.islink(dest)

    work = create_workenv(tmp_path, datadir)
    work.copy_if_needed("master.fits", src, dest)

    assert not os.path.islink(dest)
    assert is_copy_of(src, dest)


def requirements_with_same_name():

    class Input(RecipeInput):
        master1 = Requirement(DataFrameType, "first")
        master2 = Requirement(DataFrameType, "second")

    return Input(
        master1=DataFrame(filename=os.path.join("a", "master.fits")),
        master2=DataFrame(filename=os.path.join("b", "master.fits")),
    )


@pytest.mark.parametrize("action", ["copy", "link"])
def test_requirements_with_same_name(tmp_path, datadir, action):
    for _ in range(2):
        # also in a second run, with the files of the first one
        work = create_workenv(tmp_path, datadir)
        reqs = requirements_with_same_name()
        work.installfiles_stage2(reqs, action=action)

        assert reqs.master1.filename != reqs.master2.filename
        with open(os.path.join(work.workdir, reqs.master1.filename)) as fd:
            assert fd.read() == "AAAA"
        with open(os.path.join(work.workdir, reqs.master2.filename)) as fd:
            assert fd.read() == "BBBB"


def test_frame_and_requirement_with_same_name(tmp_path, datadir):
    work = create_workenv(tmp_path, datadir)

    obsres = ObservationResult()
    obsres.frames = [DataFrame(filename=os.path.join("a", "master.fits"))]
    work.installfiles_stage1(obsres, action="copy")

    class Input(RecipeInput):
        master = Requirement(DataFrameType, "a requirement")

    reqs = Input(master=DataFrame(filename=os.path.join("b", "master.fits")))
    work.installfiles_stage2(reqs, action="copy")

    frame_name = os.path.basename(obsres.frames[0].filename)
    assert frame_name == "master.fits"
    assert reqs.master.filename != frame_name
    with open(os.path.join(work.workdir, frame_name)) as fd:
        assert fd.read() == "AAAA"
    with open(os.path.join(work.workdir, reqs.master.filename)) as fd:
        assert fd.read() == "BBBB"


def test_same_file_as_frame_and_requirement(tmp_path, datadir):
    """The same file is installed once, with its name"""
    work = create_workenv(tmp_path, datadir)

    obsres = ObservationResult()
    obsres.frames = [DataFrame(filename=os.path.join("a", "master.fits"))]
    work.installfiles_stage1(obsres, action="copy")

    class Input(RecipeInput):
        master = Requirement(DataFrameType, "a requirement")

    reqs = Input(master=DataFrame(filename=os.path.join("a", "master.fits")))
    work.installfiles_stage2(reqs, action="copy")

    assert reqs.master.filename == "master.fits"
