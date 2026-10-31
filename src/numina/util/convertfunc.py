#
# Copyright 2017 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Convert strings to functions in data load"""

try:
    from functools import singledispatch
except ImportError:
    from pkgutil import simplegeneric as singledispatch

import numpy.polynomial.polynomial as nppol
from scipy.interpolate import UnivariateSpline


def json_deserial_func_u(args):
    """
    Rebuild UnivariateSpline

    Parameters
    ----------
    args

    Returns
    -------
    UnivariateSpline
    """
    # rebuild UnivariateSpline
    return UnivariateSpline._from_tck(args)


def json_deserial_func_p(args):
    """
    Rebuild Polynomial
    Parameters
    ----------
    args

    Returns
    -------
    numpy.polynomial.polynomial.Polynomial
    """
    # args is a list (from JSON) or an array (from pickle or deepcopy)
    if args is not None and len(args) > 0:
        value = nppol.Polynomial(args)
    else:
        value = nppol.Polynomial([0.0])
    return value


_json_deserial_func_map = {
    "spline1d": json_deserial_func_u,
    "polynomial": json_deserial_func_p,
    "polynomial1d": json_deserial_func_p,
}


def convert_function(node):
    """Rebuild a function from its serialized form.

    ``node`` is a dictionary with the type of function in ``function``
    (``spline1d``, ``polynomial`` or ``polynomial1d``) and its parameters in ``params``.

    ``node`` is the output of ``json_serial_function``, used directly (for
    example, in ``__setstate__`` called by ``pickle`` or ``copy.deepcopy``)
    or after a round trip through JSON, so the parameters can be numpy
    arrays or lists.
    """
    tipo = node["function"]
    args = node["params"]

    json_deserial_func = _json_deserial_func_map[tipo]
    value = json_deserial_func(args)

    return value


@singledispatch
def json_serial_function(_):
    """Return the serialized form of a function, an empty dictionary for unsupported types."""
    return {}


@json_serial_function.register(UnivariateSpline)
def json_serial_u(val):
    """Return the serialized form of a ``UnivariateSpline``, with its private ``_eval_args``."""
    serial = {}
    # This is generic and should be somewhere else
    serial["function"] = "spline1d"
    serial["params"] = val._eval_args
    return serial


@json_serial_function.register(nppol.Polynomial)
def json_serial_p(val):
    """Return the serialized form of a ``Polynomial``, with its coefficients."""
    serial = {}
    # This is generic and should be somewhere else
    serial["function"] = "polynomial1d"
    serial["params"] = val.coef
    return serial
