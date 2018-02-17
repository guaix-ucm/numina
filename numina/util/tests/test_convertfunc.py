#
# Copyright 2017 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Test convert strings to functions in data load """

import pytest

import numpy
import numpy.polynomial.polynomial as nppol

from numina.util.convertfunc import json_serial_function


def test_convert_poly1():
    coef = [1.0, 0.4, -6.1]
    poly1 = nppol.Polynomial(coef)
    result = json_serial_function(poly1)
    assert result['function'] == 'polynomial1d'
    assert numpy.allclose(result['params'], coef)
