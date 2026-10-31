import pytest

import tests.support.simpleobj as simple


@pytest.mark.result_compare
def test_result_compare_demo1():
    from numina.types.frame import DataFrameType
    from numina.types.structured import BaseStructuredCalibration

    import numina.core

    simple_img = simple.create_simple_hdul()
    simple_stc = simple.create_simple_structured()

    class BB(numina.core.RecipeResult):
        prod1 = numina.core.Result(int, "something1")
        prod2 = numina.core.Result(DataFrameType, "something2")
        prod3 = numina.core.Result(BaseStructuredCalibration, "something3")

    result = BB(prod1=1, prod2=simple_img, prod3=simple_stc)
    return result


@pytest.mark.result_compare
def test_result_compare_demo2():
    from numina.types.frame import DataFrameType
    from numina.types.structured import BaseStructuredCalibration

    import numina.core

    simple_img = simple.create_simple_hdul()
    simple_stc = simple.create_simple_structured()

    class BB(numina.core.RecipeResult):
        prod1 = numina.core.Result(int, description="something1", destination="prod4")
        prod2 = numina.core.Result(DataFrameType, "something2")
        prod3 = numina.core.Result(BaseStructuredCalibration, "something3")

    result = BB(prod1=1, prod2=simple_img, prod3=simple_stc)
    return result


def _write_fits(path, value):
    import numpy
    import astropy.io.fits as fits

    fits.PrimaryHDU(data=numpy.full((3, 3), value, dtype="float32")).writeto(path)


def _result_dirs(tmp_path, values1, values2):
    dirs = []
    for name, values in [("ref", values1), ("res", values2)]:
        d = tmp_path / name
        d.mkdir()
        for fname, value in values.items():
            _write_fits(d / fname, value)
        dirs.append(str(d))
    return dirs


def test_compare_result_dirs_identical(tmp_path):
    from numina.testing.pytest_resultcmp import compare_result_dirs

    values = {"a.fits": 1.0, "b.fits": 2.0}
    identical, msg = compare_result_dirs(*_result_dirs(tmp_path, values, values))
    assert identical
    assert msg == "ok"


def test_compare_result_dirs_all_files(tmp_path):
    """All the FITS files are compared, not only the first one (issue #264)"""
    from numina.testing.pytest_resultcmp import compare_result_dirs

    ref = {"a.fits": 1.0, "b.fits": 2.0}
    res = {"a.fits": 1.0, "b.fits": 3.0}
    identical, msg = compare_result_dirs(*_result_dirs(tmp_path, ref, res))
    assert not identical
    assert isinstance(msg, str)
    assert "b.fits" in msg


def test_compare_result_dirs_different_files(tmp_path):
    """With different files, the message is a string"""
    from numina.testing.pytest_resultcmp import compare_result_dirs

    ref = {"a.fits": 1.0, "b.fits": 2.0}
    res = {"a.fits": 1.0, "c.fits": 2.0}
    identical, msg = compare_result_dirs(*_result_dirs(tmp_path, ref, res))
    assert not identical
    assert isinstance(msg, str)
    assert "b.fits" in msg
