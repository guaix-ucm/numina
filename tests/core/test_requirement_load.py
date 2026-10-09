"""The values of the requirements in the observation result are loaded as those of the DAL"""

import numpy

import numina.store
import numina.types.datatype as dt
from numina.core import ObservationResult
from numina.core.dataholders import Requirement
from numina.types.linescatalog import LinesCatalog


def write_lines(path):
    path.write_text("4000.0 1.0\n5000.0 2.0\n")
    return str(path)


def test_lines_catalog_in_ob(tmp_path):
    lines = write_lines(tmp_path / "lines.dat")
    req = Requirement(LinesCatalog, "lines", destination="lines_catalog")
    ob = ObservationResult()
    ob.requirements = {"lines_catalog": lines}

    value = req.query_on_ob(ob)

    assert isinstance(value, numpy.ndarray)
    numpy.testing.assert_array_equal(value, numina.store.load(req.type, lines))


def test_list_of_lines_catalogs_in_ob(tmp_path):
    lines = [write_lines(tmp_path / f"lines{idx}.dat") for idx in range(2)]
    req = Requirement(dt.ListOfType(LinesCatalog), "lines", destination="lines_catalogs")
    ob = ObservationResult()
    ob.requirements = {"lines_catalogs": lines}

    value = req.query_on_ob(ob)

    assert len(value) == 2
    assert all(isinstance(v, numpy.ndarray) for v in value)


def test_dump_without_hooks():
    class Plain:
        pass

    assert numina.store.dump(Plain(), 3, "where") == 3
