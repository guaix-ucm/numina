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

__version__ = '0.5.0'

# Top level NullHandler
logging.getLogger("numina").addHandler(logging.NullHandler())

def braid(*iterables):
    '''Return the elements of each iterator in turn until some is exhausted.
    
    This function is similar to the roundrobin example 
    in itertools documentation.
    
    >>> a = iter([1,2,3,4])
    >>> b = iter(['a', 'b'])
    >>> c = iter([1,1,1,1,'a', 'c'])
    >>> d = iter([1,1,1,1,1,1])
    >>> list(braid(a, b, c, d))
    [1, 'a', 1, 1, 2, 'b', 1, 1]
    '''
    
    from itertools import izip
    
    for itbl in izip(*iterables):
        for it in itbl:
            yield it

class ReductionResult(object):
    def __init__(self):
        self.id = None
        self.reduction_block = None
        self.other = None
        self.status = 0
        self.picklable = {}

class FrameInformation(object):
    def __init__(self):
        self.label = None
        self.object = None
        self.target = None
        self.itype = None
        self.exposure = 0.0
        self.ra = 0.0
        self.dec = 0.0
        self.mdj = 0.0
        self.airmass = 1.0

class ObservingResult(object):
    def __init__(self):
        self.id = None
        self.mode = None
        self.instrument = None
        self.images = [] # list of FrameInformation
        self.children = [] # other ObservingResult
        

def frameinfo_from_list(values):
    # FIXME: modify when format is changed
    # For this format
    # [r0007.fits, M 33, 10.0, TARGET, 23.4620835, 30.66027777]
    frameinfo = FrameInformation()
    frameinfo.label = values[0]
    frameinfo.object = values[1]
    frameinfo.exposure = values[2]
    frameinfo.itype = values[3]
    frameinfo.ra = values[4]
    frameinfo.dec = values[5]
    return frameinfo

def obsres_from_dict(values):
    
    obsres = ObservingResult()
    
    obsres.id = values['id']
    obsres.mode = values['mode']
    obsres.instrument = values['instrument']
    obsres.images = [frameinfo_from_list(val) for val in values['images']]
    
    return obsres

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

