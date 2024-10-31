
import numpy
import numpy as np

import numina.array._combine as _c  # noqa


def test_error2():
    """test general functionality"""
    data1 = numpy.array([[-1, -3], [5, 0]])
    data2 = numpy.array([[1, 2], [8, 3]])
    data3 = numpy.array([[9, 1], [2, 4]])
    mask1 = numpy.array([[0, 0], [0, 0]])
    mask2 = numpy.array([[0, 0], [0, 1]])
    mask3 = numpy.array([[0, 0], [0, 1]])
    exp_res = numpy.array([[3, 0], [5, 0]])
    exp_var = numpy.array([[28, 7], [9, 0]])
    exp_pix = numpy.array([[3, 3], [3, 1]])
    a, b, c = _c.generic_combine_miter(_c.mean_method(), [data1, data2, data3], masks=[mask1, mask2, mask3])
    assert np.allclose(a, exp_res)
    assert np.allclose(b, exp_var)
    assert np.allclose(c, exp_pix)


def test_error3():
    """test general functionality"""
    data1 = numpy.array([[-1, -3], [5, 0]])
    data2 = numpy.array([[1, 2], [8, 3]])
    data3 = numpy.array([[9, 1], [2, 4]])
    mask1 = numpy.array([[0, 0], [0, 0]])
    mask2 = numpy.array([[0, 0], [0, 0]])
    mask3 = numpy.array([[0, 0], [0, 0]])
    out_res = numpy.array([[1, 2], [1, 2]])
    out_var = numpy.array([[1, 2], [1, 2]])
    out_pix = numpy.array([[1, 2], [1, 2]])
    exp_res = numpy.array([[3, 0], [5, 2]])
    exp_var = numpy.array([[28, 7], [9, 4]])
    exp_pix = numpy.array([[3, 3], [3, 3]])
    a, b, c = _c.generic_combine_miter(_c.mean_method(), [data1, data2, data3], masks=[mask1, mask2, mask3],
                                       out_res=out_res, out_var=out_var, out_pix=out_pix)
    assert np.allclose(a, exp_res)
    assert np.allclose(b, exp_var)
    assert np.allclose(c, exp_pix)


def test_error4():
    """test general functionality"""
    data1 = numpy.array([[-1, -3], [5, 0]])
    out_res = numpy.array([[1, 2], [1, 2]])
    out_var = numpy.array([[1, 2], [1, 2]])
    out_pix = numpy.array([[1, 2], [1, 2]])
    exp_res = numpy.array([[-1, -3], [5, 0]])
    exp_var = numpy.array([[0, 0], [0, 0]])
    exp_pix = 100 * numpy.array([[1, 1], [1, 1]])
    a, b, c = _c.generic_combine_miter(_c.mean_method(), [data1]*100,
                                       out_res=out_res, out_var=out_var, out_pix=out_pix)
    assert np.allclose(a, exp_res)
    assert np.allclose(b, exp_var)
    assert np.allclose(c, exp_pix)


def test_error5():
    """test general functionality"""
    data1 = numpy.array([[-1, -3], [5, 0]])
    exp_res = numpy.array([[-1, -3], [5, 0]])
    exp_var = numpy.array([[0, 0], [0, 0]])
    exp_pix = 100 * numpy.array([[1, 1], [1, 1]])
    a, b, c = _c.generic_combine_miter(_c.mean_method(), [data1]*100)
    assert np.allclose(a, exp_res)
    assert np.allclose(b, exp_var)
    assert np.allclose(c, exp_pix)


def test_error6():
    """test general functionality"""
    data1 = numpy.array([[-1, -3], [5, 0]])
    data2 = numpy.array([[2, 0], [4, 1]])
    exp_res = numpy.array([[0.5, -1.5], [4.5, 0.5]])
    exp_var = numpy.array([[4.5, 4.5], [0.5, 0.5]])
    exp_pix = 2 * numpy.array([[1, 1], [1, 1]])
    a, b, c = _c.generic_combine_miter(_c.mean_method(), [data1, data2])
    assert np.allclose(a, exp_res)
    assert np.allclose(b, exp_var)
    assert np.allclose(c, exp_pix)
