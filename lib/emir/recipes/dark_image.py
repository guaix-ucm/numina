#
# Copyright 2008-2010 Sergio Pascual
# 
# This file is part of PyEmir
# 
# PyEmir is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
#

'''Dark image recipe.

Recipe to process dark images. The dark images will be combined 
weighting with the inverses of the corresponding variance extension. 
They do not have to be of the same exposure time t, they will be 
scaled to the same t0 ~ 60s (those with very short exposure time 
should be avoided).

**Observing mode:

 * Dark current Image (3.2)

**Inputs:**

 * A list of dark images 
 * A model of the detector (gain, RN)

**Outputs:**

 * A combined dark frame, with variance extension and quality flag. 

**Procedure:**

The process will remove cosmic rays (using a typical sigma-clipping algorithm).

''' 


import logging

import pyfits

import numina.recipes as nr
from numina.array.combine import mean
from numina.exceptions import RecipeError
from emir.instrument.headers import EmirImage


_logger = logging.getLogger("emir.recipes")
        
        
class ParameterDescription(nr.ParameterDescription):
    def __init__(self):
        inputs={'images': []}
        optional={}
        super(ParameterDescription, self).__init__(inputs, optional)
        

class Result(nr.RecipeResult):
    '''Result of the DarkImaging recipe.'''
    def __init__(self, qa, dark):
        super(Result, self).__init__(qa)
        self.products['dark'] = dark

class Recipe(nr.RecipeBase):
    '''Recipe to process data taken in Dark current image Mode.
    
    Here starts the long description...
    It continues several lines
    
    '''
    def __init__(self):
        super(Recipe, self).__init__()
        # Default values. This can be read from a file
        self.creator = EmirImage()
        
    def setup(self, param):
        super(Recipe, self).setup(param)
        self.images = self.inputs['images']
        
        
    def process(self):
        fd = []
        try:
            for i in self.images:
                _logger.debug('Loading %s', i)
                fd.append(pyfits.open(i))
        except IOError, err:
            _logger.error(err)
            _logger.debug('Cleaning up hdus')
            for i in fd:
                i.close()            
            raise RecipeError(err)
        
        _logger.debug('We have %d images', len(fd))
        # Data from the primary extension
        data = [i['primary'].data for i in fd]
        #
        result = mean(data)
  
        # Final structure
        extensions = [('VARIANCE', None, None), ('NUMBER', None, None)]
        hdulist = self.creator.create(result, None, extensions)
        
        return Result(hdulist)
        
