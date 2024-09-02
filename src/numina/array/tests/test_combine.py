#
# Copyright 2008-2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import numpy
import pytest

import numina.array._combine as _c


def test_error1():
    """combine method is not valid"""
    data = numpy.array([[1, 2], [1, 2]])
    with pytest.raises(TypeError):
        _c.generic_combine("dum", [data])


def test_error2():
    """input list is empty"""
    with pytest.raises(ValueError):
        _c.generic_combine(_c.mean_method(), [])


def test_error3():
    """images don't have the same shape"""
    data = numpy.array([[1, 2], [1, 2]])
    databig = numpy.zeros((256, 256))
    with pytest.raises(ValueError):
        _c.generic_combine(_c.mean_method(), [data, databig])


def test_error4():
    """incorrect number of masks"""
    data = numpy.array([[1, 2], [1, 2]])
    mask = numpy.array([[True, False], [True, False]])
    with pytest.raises(ValueError):
        _c.generic_combine(_c.mean_method(), [data], masks=[mask, mask])


def test_error5():
    """mask and image have different shape"""
    data = numpy.array([[1, 2], [1, 2]])
    masks = [numpy.array([True, False, True])]
    with pytest.raises(ValueError):
        _c.generic_combine(_c.mean_method(), [data], masks=masks)


def test_error6():
    """output has wrong shape"""
    data = numpy.array([[1, 2], [1, 2]])
    out = numpy.empty((3, 2, 239))
    with pytest.raises(ValueError):
        _c.generic_combine(_c.mean_method(), [data], out_res=out[0], out_var=out[1], out_pix=out[2])


@pytest.mark.parametrize("bad_par", [
    [[1, 2]],
    [1, 2]
])
def test_scales_error(bad_par):
    data = numpy.array([[1, 2], [1, 2]])

    with pytest.raises(ValueError):
        _c.generic_combine(_c.mean_method(), [data], scales=bad_par)


@pytest.mark.parametrize("bad_par", [
    [[1, 2]],
    [1, 2]
])
def test_zeros_error(bad_par):
    data = numpy.array([[1, 2], [1, 2]])
    with pytest.raises(ValueError):
        _c.generic_combine(_c.mean_method(), [data], zeros=bad_par)


@pytest.mark.parametrize("bad_par", [
    [[1, 2]],
    [1, 2]
])
def test_weights_error(bad_par):
    data = numpy.array([[1, 2], [1, 2]])
    with pytest.raises(ValueError):
        _c.generic_combine(_c.mean_method(), [data], weights=bad_par)


def test_combine_mask_average():
    """Average combine: combination of integer arrays with masks."""
    input1 = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
    input2 = numpy.array([[3, 2, 8, 4], [6, 2, 0, 4], [1, 3, 3, 4]])
    input3 = numpy.array([[7, 2, 1, 4], [1, 2, 0, 4], [44, 2, 2, 0]])
    inputs = [input1, input2, input3]

    mask1 = numpy.array([[False, False, False, True],
                         [False, True, False, False],
                         [False, True, False, False]])
    mask2 = numpy.array([[False, False, False, False],
                         [False, True, False, False],
                         [False, True, False, False]])
    mask3 = numpy.array([[False, False, False, False],
                         [False, True, False, False],
                         [False, False, True, False]])
    masks = [mask1, mask2, mask3]
    rres = numpy.array([[3.66666667, 2., 4., 4.],
                        [2.6666666666666665, 0., 1., 4.],
                        [18., 2., 1.5, 2.66666667]])
    rvar = numpy.array([[3 * 3.11111111, 0., 3 * 4.33333333, 0.],
                        [3 * 2.77777778, 0., 3 * 1.,  0.],
                        [3 * 174.33333333, 0., 2 * 2.25, 3 * 1.77777778]
                        ])

    rnum = numpy.array([[3, 3, 3, 2],
                        [3, 0, 3, 3],
                        [3, 1, 2, 3]])

    out0, out1, out2 = _c.generic_combine(_c.mean_method(), inputs, masks=masks)
    assert numpy.allclose(out0, rres)
    assert numpy.allclose(out1, rvar)
    assert numpy.allclose(out2, rnum)


def test_combine_average():
    """Average combine: combination of integer arrays."""
    input1 = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
    input2 = numpy.array([[3, 2, 8, 4], [6, 2, 0, 4], [1, 3, 3, 4]])
    input3 = numpy.array([[7, 2, 1, 4], [1, 2, 0, 4], [44, 2, 2, 0]])
    inputs = [input1, input2, input3]

    # Results
    rres = numpy.array([[3.66666667, 2., 4., 4.0],
                        [2.6666666666666665, 2., 1., 4.],
                        [18., 2.33333333, 1.666666667, 2.66666667]])
    rvar = 3 * numpy.array([[9.3333333333333339, 0., 13.0, 0.],
                            [8.3333333333333339, 0., 3.00000000, 0.],
                            [523.0, 0.33333333333333337, 2.333333333333333,
                             5.3333333333333339]]) / len(inputs)
    rnum = numpy.array([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]])

    out0, out1, out2 = _c.generic_combine(_c.mean_method(), inputs)
    assert numpy.allclose(out0, rres)
    assert numpy.allclose(out1, rvar)
    assert numpy.allclose(out2, rnum)


def test_combine_sum():
    """Average combine: combination of integer arrays."""
    # Inputs
    input1 = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
    input2 = numpy.array([[2, 2, 8, 4], [6, 2, 0, 4], [1, 3, 3, 4]])
    input3 = numpy.array([[1, 2, 1, 4], [1, 2, 0, 4], [44, 2, 2, 0]])
    inputs = [input1, input2, input3]
    # Results
    rres = numpy.array([[4, 6, 12, 12],
                        [8, 6, 3, 12],
                        [54, 7, 5, 8]])
    # The variance result is not useful

    rvar = numpy.array([[1, 0, 39, 0],
                        [25, 0, 9, 0],
                        [1569, 1, 7, 16]]) * 3
    rnum = numpy.array([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]])

    out0, out1, out2 = _c.generic_combine(_c.sum_method(), inputs)
    assert numpy.allclose(out0, rres)
    assert numpy.allclose(out1, rvar)
    assert numpy.allclose(out2, rnum)


def test_combine_median1():
    """Median combine: combination of integer arrays."""
    # Inputs
    input1 = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
    input2 = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
    input3 = numpy.array([[7, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
    input4 = numpy.array([[7, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
    input5 = numpy.array([[7, 2, 1, 4], [1, 2, 0, 4], [44, 2, 2, 0]])
    inputs = [input1, input2, input3, input4, input5]

    rres = input3
    rvar = [[16.954474097331239, 0.0, 1.2558869701726849, 0.0],
            [0.0, 0.0, 2.8257456828885403, 0.0],
            [384.61538461538458, 0.0, 1.2558869701726847,
            5.0235478806907397]]

    out0, out1, out2 = _c.generic_combine(_c.median_method(), inputs)

    assert numpy.allclose(out0, rres)
    assert numpy.allclose(out1, rvar)
    assert numpy.allclose(out2, len(inputs))


def test_combine_median2():
    """Median combine: combination an even number of integer arrays."""
    # Inputs
    input1 = numpy.array([[1, 2, 3, -4]])
    input2 = numpy.array([[1, 2, 6, 4]])
    input3 = numpy.array([[7, 3, 8, -4]])
    input4 = numpy.array([[7, 2, 3, 4]])
    inputs = [input1, input2, input3, input4]

    rres = numpy.array([[4, 2, 4.5, 0.0]], dtype='float')
    rvar = [18.838304552590266, 0.39246467817896391, 9.419152276295133,
            33.490319204604916]

    out0, out1, out2 = _c.generic_combine(_c.median_method(), inputs)

    assert numpy.allclose(out0, rres)
    assert numpy.allclose(out1, rvar)
    assert numpy.allclose(out2, len(inputs))
