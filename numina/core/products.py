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

'''
Basic Data Products
'''

from astropy.io import fits
from .pipeline import InstrumentConfiguration
from .oresult import ObservationResult
from .dataframe import DataFrame
from numina.qc import QC
from .types import DataType

class DataProduct(DataType):

    def __init__(self, ptype, default=None):
        super(DataProduct, self).__init__(ptype, default=default)

    def suggest(self, obj, suggestion):
        return obj

    def __repr__(self):
        sclass = type(self).__name__
        return "%s()" % (sclass, )

class FrameDataProduct(DataProduct):
    def __init__(self):
        super(DataProduct, self).__init__(DataFrame)

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
            raise TypeError('object of type %r cannot be converted to DataFrame')

    def validate(self, obj):
        if isinstance(obj, basestring):
            # check that this is a FITS file
            # try - open
            # FIXME
            pass
        elif isinstance(obj, fits.HDUList):
            # is an HDUList
            pass
        elif isinstance(obj, fits.PrimaryHDU):
            # is an HDUList
            pass
        elif isinstance(obj, DataFrame):
            #is a DataFrame
            return True
        else:
            raise TypeError('%r is not a valid FrameDataProduct' % obj)

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

class ObservationResultType(DataType):
    '''The type of ObservationResult.'''
    
    def __init__(self):
        super(ObservationResultType, self).__init__(ptype=ObservationResult)

class InstrumentConfigurationType(DataType):
    '''The type of InstrumentConfiguration.'''
    
    def __init__(self):
        super(InstrumentConfigurationType, self).__init__(ptype=InstrumentConfiguration)

class QualityControlProduct(DataProduct):
    def __init__(self):
        super(QualityControlProduct, self).__init__(ptype=QC, 
            default=QC.UNKNOWN)

class NonLinearityPolynomial(list):
    def __init__(self, *args, **kwds):
        super(NonLinearityPolynomial, self).__init__(self, *args, **kwds)

class NonLinearityProduct(DataProduct):
    def __init__(self, default=[1.0, 0.0]):
        super(NonLinearityProduct, self).__init__(ptype=NonLinearityPolynomial,                 default=default)

class Centroid2D(object):
    '''Temptative Centroid Class.'''
    def __init__(self):
        self.centroid = [[0.0, 0.0], [0.0, 0.0]]
        self.sigma = [[0.0, 0.0], [0.0, 0.0]]
        self.flux = []
        
