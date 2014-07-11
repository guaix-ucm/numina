#
# Copyright 2008-2014 Universidad Complutense de Madrid
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

import sys


import numpy
from astropy.io import fits

from .qc import QC
from .pipeline import InstrumentConfiguration
from .pipeline import import_object
from .oresult import ObservationResult
from .dataframe import DataFrame
from .types import DataType
from .typedialect import dialect_info
from numina.frame.schema import Schema
from .validation import ValidationError


class DataProductType(DataType):
    def __init__(self, ptype, default=None):
        super(DataProductType, self).__init__(ptype, default=default)
        self.dialect = dialect_info(self)

    def suggest(self, obj, suggestion):
        return obj

    def __repr__(self):
        sclass = type(self).__name__
        return "%s()" % (sclass, )

_base_schema = {
    'keywords': {
        'INSTRUME': {'valid': True},
        'READMODE': {'valid': True},
        'EXPTIME': {'value': float},
        'NUMINAID': {'value': int}
        }
    }

class DataFrameType(DataProductType):
    def __init__(self):
        super(DataFrameType, self).__init__(DataFrame)
        self.headerschema = Schema(_base_schema)

    def store(self, obj):
        # We accept None representing No Image
        if obj is None:
            return None
        elif isinstance(obj, basestring):
            return DataFrame(filename=obj)
        elif isinstance(obj, DataFrame):
            return obj
        elif isinstance(obj, fits.HDUList):
            return DataFrame(frame=obj)
        elif isinstance(obj, fits.PrimaryHDU):
            return DataFrame(frame=fits.HDUList([obj]))
        else:
            raise TypeError('object of type %r cannot be converted to DataFrame' % obj)

    def validate(self, value):
        # obj can be None or a DataFrame
        if value is None:
            return True
        else:
            try:
                with value.open() as hdulist:
                    self.validate_hdulist(hdulist)
            except StandardError as err:
                raise ValidationError, err, sys.exc_info()[2]

    def validate_hdulist(self, hdulist):
        pass

    def suggest(self, obj, suggestion):
        if not isinstance(suggestion, basestring):
            raise TypeError('suggestion must be a string, not %r' % suggestion)
            return obj
        if isinstance(obj, basestring):
            # check that this is a FITS file
            # try - open
            # FIXME
            pass
        elif isinstance(obj, fits.HDUList):
            obj[0].update('filename', suggestion) 
        elif isinstance(obj, DataFrame):
            obj.filename = suggestion
        return obj

class ArrayType(DataProductType):
    def __init__(self, default=None):
        super(ArrayType, self).__init__(ptype=numpy.ndarray, default=default)


    def store(self, obj):
        return self.store_as_array(obj)
        
    def store_as_array(self, obj):
        result = numpy.array(obj)
        return result

# FIXME: this is hack, thus should be provided by DRPS
def _gimme_validator_for(instrument, mode):
    validators = {'EMIR': 
            {'IMAGE_BIAS': 'emir.dataproducts.RawBias',
             'IMAGE_DARK': 'emir.dataproducts.RawDark',
            'INTENSITY_FLAT_FIELD': 'emir.dataproducts.RawIntensityFlat'
             }
           }
    
    if instrument not in validators:
        return DataFrameType
    else:
        modes = validators[instrument]
        if mode not in modes:
            return DataFrameType
        else:
            fqn = modes[mode]
            return import_object(fqn)
    return DataFrameType

class ObservationResultType(DataType):
    '''The type of ObservationResult.'''
    
    def __init__(self):
        super(ObservationResultType, self).__init__(ptype=ObservationResult)
        
    def validate(self, obj):
        RawType = _gimme_validator_for(obj.instrument, obj.mode) 
        imgtype = RawType()
        for f in obj.frames:
            imgtype.validate(f)
        return True

class InstrumentConfigurationType(DataType):
    '''The type of InstrumentConfiguration.'''
    
    def __init__(self):
        super(InstrumentConfigurationType, self).__init__(ptype=InstrumentConfiguration)
        
    def validate(self, obj):
        return True

class QualityControlProduct(DataProductType):
    def __init__(self):
        super(QualityControlProduct, self).__init__(ptype=QC, 
            default=QC.UNKNOWN)

#class NonLinearityPolynomial(list):
#    def __init__(self, *args, **kwds):
#        super(NonLinearityPolynomial, self).__init__(self, *args, **kwds)

#class NonLinearityProduct(DataProduct):
#    def __init__(self, default=[1.0, 0.0]):
#        super(NonLinearityProduct, self).__init__(ptype=NonLinearityPolynomial, 
#                                                  default=default)

#class Centroid2D(object):
#    '''Temptative Centroid Class.'''
#    def __init__(self):
#        self.centroid = [[0.0, 0.0], [0.0, 0.0]]
#        self.sigma = [[0.0, 0.0], [0.0, 0.0]]
#        self.flux = []
        










