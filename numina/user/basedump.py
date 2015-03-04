from __future__ import print_function

import warnings

import numpy
import yaml

from numina.core import Product
from numina.core import RecipeResult
from numina.core import RecipeRequirements
from numina.core import ErrorRecipeResult
from numina.core import DataFrameType, DataProductType
from numina.core.types import PlainPythonType
from numina.core.products import ArrayType
from numina.core import DataFrame

from .helpers import ProcessingTask
from .dump import dump

# These should go to megara
from megaradrp.products import TraceMapType as TraceMap
from megaradrp.products import MasterBias


@dump.register(ProcessingTask)
def _(tag, obj, where):
    result = tag.result
    tag.result = dump(result, result, where)
    with open(where.task, 'w+') as fd:
        yaml.dump(tag.__dict__, fd)
    return where.task

@dump.register(ErrorRecipeResult)
def _(tag, obj, where):
    with open(where.result, 'w+') as fd:
        yaml.dump(result, fd)

    return where.result

@dump.register(RecipeResult)
def _(tag, obj, where):

    saveres = {}
    for key in obj.__class__:
        pc = obj.__class__[key]
        val = getattr(obj, key)
        #_logger.debug('dump %r of type %r', val, type(val))
        saveres[key] = dump(pc, val, where)

    with open(where.result, 'w+') as fd:
        yaml.dump(saveres, fd)

    return where.result


@dump.register(Product)
def _(tag, obj, where):
    where.destination = tag.dest
    return dump(tag.type, obj, where)


@dump.register(DataProductType)
def _(tag, obj, where):    
    return obj


@dump.register(PlainPythonType)
def _(tag, obj, where):    
    return obj

@dump.register(ArrayType)
def _(tag, obj, where):    
    return dump.registry[numpy.ndarray](tag, obj, where)

from emir.dataproducts import ArrayType

@dump.register(ArrayType)
def _(tag, obj, where):    
    return dump.registry[numpy.ndarray](tag, obj, where)

@dump.register(DataFrameType)
def _(tag, obj, where):    
    return dump.registry[DataFrame](tag, obj, where)


@dump.register(DataFrame)
def _(tag, obj, where):
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


@dump.register(TraceMap)
def _(tag, obj, where):

    filename = where.destination + '.yaml'

    with open(filename, 'w') as fd:
        yaml.dump(fd, obj)

    return filename


@dump.register(numpy.ndarray)
def _(tag, obj, where):
    # FIXME:
    #filename = where.get_next_basename('.txt')
    filename = where.destination + '.txt'
    numpy.savetxt(filename, obj)
    return filename


@dump.register(list)
def _(tag, obj, where):
    return [dump(tag, o, where) for o in obj]

