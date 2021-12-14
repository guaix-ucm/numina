
import os.path

from ..helpers import WorkEnvironment


def test_work1(tmpdir):
    """Test default definitions"""
    obsid = 100
    base = 'base'
    basedir = str(tmpdir.dirpath(base))

    work = WorkEnvironment(obsid, basedir=basedir)
    work.sane_work()

    assert work.workdir == os.path.join(basedir, f"obsid{obsid}_work")
    assert work.basedir == basedir
    assert work.resultsdir == os.path.join(basedir, f"obsid{obsid}_results")

    index_base = "index.pkl"
    assert work.index_file == os.path.join(work.workdir, index_base)

    assert os.path.isdir(work.workdir)
    assert os.path.isdir(work.resultsdir)
    assert os.path.isdir(work.basedir)
    assert os.path.isfile(work.index_file)

