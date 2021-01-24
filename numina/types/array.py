# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import numpy

from .datatype import DataType


class ArrayType(DataType):
    """A type of array."""
    def __init__(self, fmt='%.18e', default=None):
        super(ArrayType, self).__init__(ptype=numpy.ndarray, default=default)
        self.fmt = fmt

    def convert(self, obj):
        return self.convert_to_array(obj)

    def convert_to_array(self, obj):
        return numpy.array(obj)

    def _datatype_dump(self, obj, where):
        return dump_numpy_array(obj, where, self.fmt)

    def _datatype_load(self, obj):
        if isinstance(obj, str):
            # if is a string, it may be a pathname, try to load it

            # heuristics, by extension
            if obj.endswith('.csv'):
                # try to open as a CSV file
                res = numpy.loadtxt(obj, delimiter=',')
            else:
                res = numpy.loadtxt(obj)
        else:
            res = obj
        return res


class ArrayNType(ArrayType):
    def __init__(self, dimensions, default=None):
        super(ArrayNType, self).__init__(default=default)
        self.N = dimensions


def get_filename_obj(hdul, hint, ext):

    if isinstance(hint, str):
        filename = hint + ext
    elif callable(hint):
        filename = hint(hdul)
    else:
        raise ValueError('"hint" is neither string nor callable')
    return filename


def get_filename_numpy_array(obj, where):
    filename = get_filename_obj(obj, where, '.txt')
    return filename


def dump_numpy_array(obj, where, fmt='%.18e'):
    filename = get_filename_numpy_array(obj, where)
    numpy.savetxt(filename, obj, fmt=fmt)
    return filename
