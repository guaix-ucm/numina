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


import numina.array.combine as c
import numina.array._combine as intl  # noqa


def test_crmean_raises1():
    with pytest.raises(ValueError):
        intl.crmean_method(-1, 1, 1)


def test_crmean_raises2():
    with pytest.raises(ValueError):
        intl.crmean_method(1, -1, 1)


def test_crmean_raises3():
    with pytest.raises(ValueError):
        intl.crmean_method(1, 1, -1)



def test_less3():
    data1 = numpy.array([[-1, -3], [5, 0]])
    data2 = numpy.array([[2, 0], [4, 1]])
    res_val = numpy.zeros_like(data1)
    res_var = res_val
    res_num = res_val
    val, var, num = c.crmean([data1, data2])
    assert numpy.allclose(val, res_val)
    assert numpy.allclose(var, res_var)
    assert numpy.allclose(num, res_num)
