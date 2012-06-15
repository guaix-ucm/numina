#
# Copyright 2008-2012 Universidad Complutense de Madrid
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

import pyfits

from .dataframe import DataFrame

class ValidationError(Exception):
    pass

class DataProduct(object):
    def validate(self, obj):
        return True

    def store(self, obj):
        return obj

class FrameDataProduct(DataProduct):

    def store(self, obj):

        if isinstance(obj, basestring):
            return DataFrame(filename=obj)
        elif isinstance(obj, DataFrame):
            return obj
        else:
            return DataFrame(frame=obj)

    def validate(self, obj):

        if isinstance(obj, basestring):
            # check that this is a FITS file
            # try - open
            pass
        elif isinstance(obj, pyfits.HDUList):
            # is an HDUList
            pass
        elif isinstance(obj, DataFrame):
            #is a DataFrame
            pass
        else:
            raise ValidationError('object is not a valid FrameDataProduct')
