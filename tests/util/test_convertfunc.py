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

from numina.util.convertfunc import convert_function, json_serial_function
from numina.util.jsonencoder import ExtEncoder


def test_convert_poly1():
    coef = [1.0, 0.4, -6.1]
    poly1 = nppol.Polynomial(coef)
    result = json_serial_function(poly1)
    assert result["function"] == "polynomial1d"
    assert numpy.allclose(result["params"], coef)


def test_convert_poly1_roundtrip():
    """A polynomial saved in JSON can be loaded (issue #265)"""
    coef = [1.0, 0.4, -6.1]
    poly1 = nppol.Polynomial(coef)
    node = json.loads(json.dumps(json_serial_function(poly1), cls=ExtEncoder))
    result = convert_function(node)
    assert isinstance(result, nppol.Polynomial)
    assert numpy.allclose(result.coef, coef)
