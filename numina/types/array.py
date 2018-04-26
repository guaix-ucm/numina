# Copyright 2008-2017 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import numpy
import six

from .datatype import DataType


class ArrayType(DataType):
    """A type of array."""
    def __init__(self, default=None):
        super(ArrayType, self).__init__(ptype=numpy.ndarray, default=default)

    def convert(self, obj):
        return self.convert_to_array(obj)

    def convert_to_array(self, obj):
        return numpy.array(obj)

    def _datatype_dump(self, obj, where):
        return dump_numpy_array(obj, where)

    def _datatype_load(self, obj):
        if isinstance(obj, six.string_types):
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


def dump_numpy_array(obj, where):
    # FIXME:
    #filename = where.get_next_basename('.txt')
    filename = where.destination + '.txt'
    numpy.savetxt(filename, obj)
    return filename
