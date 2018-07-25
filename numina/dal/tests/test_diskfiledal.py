
from ..diskfiledal import FileFinder
from ..diskfiledal import FileFinderGTC


def test_candidates_ff(tmpdir):

    subd = ['sub1', 'sub2', 'sub3']
    for s in subd:
        tmpdir.mkdir(s)

    m = FileFinder()
    ca = m.candidates(str(tmpdir))
    assert sorted(ca) == sorted(subd)


def test_check_ff(tmpdir):

    dirname = 'sub1'
    fname1 = "hello.txt"
    fname2 = ".hidden"
    fname3 = "dir"

    newdir = tmpdir.mkdir(dirname)
    newfile1 = newdir.join(fname1)
    newfile1.write('content')

    newfile2 = newdir.join(fname2)
    newfile2.write('content')

    newdir.mkdir(fname3)

    m = FileFinder()
    # Conversion is needed only in Python < 3.6
    valid = m.check(str(newdir), fname1)
    assert valid

    invalid = m.check(str(newdir), fname2)
    assert invalid == False

    invalid = m.check(str(newdir), fname2)
    assert invalid == False


def test_candidates_ffg(tmpdir):

    subd = ['sub1', 'sub2', 'sub3']
    res = [('result.json', 1), ('sub1', 0), ('sub2', 0), ('sub3', 0)]
    for s in subd:
        tmpdir.mkdir(s)

    m = FileFinderGTC()
    ca = m.candidates(str(tmpdir))
    assert sorted(ca) == sorted(res)
