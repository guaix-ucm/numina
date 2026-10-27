import pytest

from numina.dal.diskfiledal import DiskFileDAL, FileFinder, FileFinderGTC, _combinations, build_product_path
from numina.exceptions import NoResultFound


def test_candidates_ff(tmpdir):

    subd = ["sub1", "sub2", "sub3"]
    for s in subd:
        tmpdir.mkdir(s)

    m = FileFinder()
    ca = m.candidates(str(tmpdir))
    assert sorted(ca) == sorted(subd)


def test_check_ff(tmpdir):

    dirname = "sub1"
    fname1 = "hello.txt"
    fname2 = ".hidden"
    fname3 = "dir"

    newdir = tmpdir.mkdir(dirname)
    newfile1 = newdir.join(fname1)
    newfile1.write("content")

    newfile2 = newdir.join(fname2)
    newfile2.write("content")

    newdir.mkdir(fname3)

    m = FileFinder()
    # Conversion is needed only in Python < 3.6
    valid = m.check(str(newdir), fname1)
    assert valid

    invalid = m.check(str(newdir), fname2)
    assert invalid is False

    invalid = m.check(str(newdir), fname2)
    assert invalid is False


def test_candidates_ffg(tmpdir):

    subd = ["sub1", "sub2", "sub3"]
    res = [("result.json", 1), ("sub1", 0), ("sub2", 0), ("sub3", 0)]
    for s in subd:
        tmpdir.mkdir(s)

    m = FileFinderGTC()
    ca = m.candidates(str(tmpdir))
    assert sorted(ca) == sorted(res)


class Provides:
    alias = "Alias"


class FakeDrp:
    def __init__(self, provided=()):
        self.provided = provided

    def query_provides(self, product):
        if product in self.provided:
            return Provides()
        raise ValueError(f"no mode provides {product}")


class Product:
    """A type that loads the content of the file"""

    @staticmethod
    def name():
        return "ProductName"

    def _datatype_load(self, obj):
        with open(obj) as fd:
            return fd.read()


class FakeOb:
    instrument = "INS"
    tags = {"filter": "J", "grism": "A"}


def make_file(path, content="content"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_combinations():
    assert list(_combinations(["a", "b"])) == [("a", "b"), ("a",), ("b",), ()]


def test_check_ffg(tmp_path):
    make_file(tmp_path / "result.json")
    m = FileFinderGTC()
    assert m.check(str(tmp_path), ("result.json", 1))
    assert not m.check(str(tmp_path), ("other.json", 0))
    assert m.build_final_path(str(tmp_path), ("result.json", 1)) == (str(tmp_path / "result.json"), 1)


def test_build_product_path(tmp_path):
    base = tmp_path / "INS" / "default"
    make_file(base / "ProductName" / "J" / "calib.json")
    make_file(base / "ProductName" / "calib.json")

    # the most specific directory first, the type has no mode
    path = build_product_path(FakeDrp(), str(tmp_path), "default", "name", Product(), FakeOb())
    assert path == (str(base / "ProductName" / "J" / "calib.json"), 0)

    make_file(base / "ProductName" / "J" / "result.json")
    path = build_product_path(FakeDrp(), str(tmp_path), "default", "name", Product(), FakeOb())
    assert path == (str(base / "ProductName" / "J" / "result.json"), 1)


def test_build_product_path_alias(tmp_path):
    make_file(tmp_path / "INS" / "conf" / "Alias" / "calib.json")
    path = build_product_path(FakeDrp([Product]), str(tmp_path), "conf", "name", Product(), FakeOb(), cls=FileFinder)
    assert path == str(tmp_path / "INS" / "conf" / "Alias" / "calib.json")


def test_build_product_path_not_found(tmp_path):
    make_file(tmp_path / "INS" / "default" / "Other" / "calib.json")
    with pytest.raises(NoResultFound, match="type ProductName compatible with tags"):
        build_product_path(FakeDrp(), str(tmp_path), "default", "name", Product(), FakeOb())


def test_disk_file_dal(tmp_path):
    base = tmp_path / "INS" / "default"
    make_file(base / "Product" / "J" / "A" / "calib.json", "first")
    (base / "Product" / "J" / "A" / "subdir").mkdir()
    make_file(base / "Product" / "J" / "calib.json", "second")

    dal = DiskFileDAL(FakeDrp(), str(tmp_path))
    result = dal.search_product("name", Product(), FakeOb())

    assert result.content == "first"
    assert result.tags == FakeOb.tags


def test_disk_file_dal_alias(tmp_path):
    make_file(tmp_path / "INS" / "default" / "Alias" / "calib.json", "aliased")
    dal = DiskFileDAL(FakeDrp([Product]), str(tmp_path))
    assert dal.search_product("name", Product(), FakeOb()).content == "aliased"


def test_disk_file_dal_id(tmp_path):
    directory = tmp_path / "INS" / "default" / "Product"
    make_file(directory / "calib.json", "first")
    (directory / "subdir").mkdir()

    dal = DiskFileDAL(FakeDrp(), str(tmp_path))
    assert dal.search_product("name", Product(), FakeOb()).id == "calib.json"


def test_disk_file_dal_not_found(tmp_path):
    dal = DiskFileDAL(FakeDrp(), str(tmp_path))
    with pytest.raises(NoResultFound):
        dal.search_product("name", Product(), FakeOb())
