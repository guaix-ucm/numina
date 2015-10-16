#
# Copyright 2008-2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

"""Register basic types with dump."""

from __future__ import print_function

import warnings

import numpy
import yaml

from numina.core import RecipeResult
from numina.core import ErrorRecipeResult
from numina.core import DataFrameType, DataProductType
from numina.core.types import PlainPythonType
from numina.core.products import ArrayType
from numina.core import DataFrame

from .dump import dump


@dump.register(ErrorRecipeResult)
def _(tag, obj, where):
    with open(where.result, 'w+') as fd:
        yaml.dump(where.result, fd)

    return where.result


@dump.register(RecipeResult)
def _(tag, obj, where):

    saveres = {}
    for key, pc in obj.stored().items():
        val = getattr(obj, key)
        where.destination = pc.dest
        saveres[key] = dump(pc.type, val, where)

    with open(where.result, 'w+') as fd:
        yaml.dump(saveres, fd)

    return where.result


@dump.register(DataProductType)
def _(tag, obj, where):
    return obj


@dump.register(PlainPythonType)
def _(tag, obj, where):    
    return obj


@dump.register(ArrayType)
def _(tag, obj, where):
    return dump_numpy_array(obj, where)


@dump.register(DataFrameType)
def _(tag, obj, where):
    return dump_dataframe(obj, where)


def dump_dataframe(obj, where):
    # save fits file
    if obj.frame is None:
        # assume filename contains a FITS file
        return None
    else:
        if obj.filename:
            filename = obj.filename
        elif 'FILENAME' in obj.frame[0].header:
            filename = obj.frame[0].header['FILENAME']
        elif hasattr(where, 'destination'):
            filename = where.destination + '.fits'
        else:
            filename = where.get_next_basename('.fits')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            obj.frame.writeto(filename, clobber=True)
        return filename


def dump_numpy_array(obj, where):
    # FIXME:
    #filename = where.get_next_basename('.txt')
    filename = where.destination + '.txt'
    numpy.savetxt(filename, obj)
    return filename


dump.register(numpy.ndarray, dump_numpy_array)

dump.register(DataFrame, dump_dataframe)


@dump.register(list)
def _(tag, obj, where):
    return [dump(tag, o, where) for o in obj]


# FIXME: this is very convoluted
from .defaultl import load_cli_storage as other


def load_cli_storage():
    return other()
