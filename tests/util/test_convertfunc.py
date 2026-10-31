#
# Copyright 2017-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Test convert strings to functions in data load"""

import json

import numpy
import numpy.polynomial.polynomial as nppol
import pytest
from scipy.interpolate import UnivariateSpline

from numina.util.convertfunc import convert_function, json_serial_function
from numina.util.jsonencoder import ExtEncoder


def test_convert_poly1():
    coef = [1.0, 0.4, -6.1]
    poly1 = nppol.Polynomial(coef)
    result = json_serial_function(poly1)
    assert result["function"] == "polynomial1d"
    assert numpy.allclose(result["params"], coef)


def _direct(node):
    """The serialized form used directly, as in pickle or deepcopy"""
    return node


def _through_json(node):
    """The serialized form saved in JSON and loaded again"""
    return json.loads(json.dumps(node, cls=ExtEncoder))


def _make_poly():
    return nppol.Polynomial([1.0, 0.4, -6.1])


def _make_spline():
    x = numpy.arange(10.0)
    return UnivariateSpline(x, x**2, k=3)


@pytest.mark.parametrize("transfer", [_direct, _through_json])
@pytest.mark.parametrize("make_func", [_make_poly, _make_spline])
def test_convert_roundtrip(make_func, transfer):
    """A serialized function can be rebuilt, with arrays or with lists (issue #265)"""
    func = make_func()
    result = convert_function(transfer(json_serial_function(func)))
    # the spline is rebuilt as UnivariateSpline, the base class of the original
    assert isinstance(func, type(result))
    x = numpy.linspace(0, 9, 20)
    assert numpy.allclose(result(x), func(x))


def test_convert_poly_empty():
    """A polynomial without coefficients is the zero polynomial"""
    result = convert_function({"function": "polynomial", "params": []})
    assert numpy.allclose(result.coef, [0.0])
