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
from .dataframe import DataFrame
from numina.qc import QC

class ValidationError(Exception):
    pass

class DataType(object):
    '''Base class for requirement types.'''
    
    def validate(self, obj):
        return True

class DataProduct(DataType):

    def __init__(self, default=None):
        self.default = default

    def store(self, obj):
        return obj

    def suggest(self, obj, suggestion):
        return obj

    def __repr__(self):
        sclass = type(self).__name__
        return "%s()" % (sclass, )

class FrameDataProduct(DataProduct):

    def store(self, obj):

        if obj is None:
            return None
        elif isinstance(obj, basestring):
            return DataFrame(filename=obj)
        elif isinstance(obj, DataFrame):
            return obj
        else:
            return DataFrame(frame=obj)

    def validate(self, obj):

        if isinstance(obj, basestring):
            # check that this is a FITS file
            # try - open
            # FIXME
            pass
        elif isinstance(obj, fits.HDUList):
            # is an HDUList
            pass
        elif isinstance(obj, DataFrame):
            #is a DataFrame
            pass
        else:
            raise ValidationError('%r is not a valid FrameDataProduct' % obj)

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

class InstrumentConfigurationType(DataType):
    '''The type of InstrumentConfiguration.'''
    
    def validate(self, value):
        if not isinstance(value, InstrumentConfiguration):
            raise ValidationError('%r is not an instance of InstrumentConfiguration')


class QualityControlProduct(DataProduct):
    def __init__(self):
        super(QualityControlProduct, self).__init__(default=QC.UNKNOWN)

    def validate(self, obj):
        if obj not in [QC.GOOD, QC.PARTIAL, QC.BAD, QC.UNKNOWN]:
            raise ValidationError('%r is not a valid QualityControlProduct' % obj)

