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

'''Numina data processing system.'''

import logging
import json

import pyfits

from numina.recipes import RecipeBase, DataFrame
from numina.recipes import obsres_from_dict

__version__ = '0.5.0'

# Top level NullHandler
logging.getLogger("numina").addHandler(logging.NullHandler())


# FIXME: pyfits.core.HDUList is treated like a list
# each extension is stored separately
class ProductEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pyfits.core.PrimaryHDU):
            filename = 'result.fits'
            if obj.header.has_key('FILENAME'):
                filename = obj.header['FILENAME']
            obj.writeto(filename, clobber=True)
            return filename
        return json.JSONEncoder.default(self, obj)

