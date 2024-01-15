
import os.path

from ..helpers import BaseWorkEnvironment


def test_work1(tmpdir):
    """Test default definitions"""
    base = 'base'
    basedir = str(tmpdir.dirpath(base))
    data = 'data'
    workdir = 'a'
    resultsdir = 'b'
    work = BaseWorkEnvironment(data, basedir, workdir, resultsdir)
    work.sane_work()

    assert work.workdir == os.path.join(basedir, workdir)
    assert work.basedir == basedir
    assert work.resultsdir == os.path.join(basedir, resultsdir)

    index_base = "index.pkl"
    assert work.index_file == os.path.join(work.workdir, index_base)

    assert os.path.isdir(work.workdir)
    assert os.path.isdir(work.resultsdir)
    assert os.path.isdir(work.basedir)
    assert os.path.isfile(work.index_file)
