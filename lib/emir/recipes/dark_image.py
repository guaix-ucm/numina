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

**Observing mode:**

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

from numina.recipes import RecipeBase
from numina.array.combine import mean
from emir.dataproducts import create_result
import numina.qa

_logger = logging.getLogger("emir.recipes")

class Recipe(RecipeBase):
    '''Recipe to process data taken in Dark current image Mode.'''
    
    capabilities = ['dark_current_image']
    
    required_parameters = [
        'nthreads',
        'images',
    ]    
        
    def __init__(self, value):
        super(Recipe, self).__init__(value)
        # Default values. This can be read from a file        
        
    def run(self):
        
        alldata = []
        allmasks = []
        
        try:
            for n in self.values['images']:
                n.open(mode='readonly', memmap=True)
                alldata.append(n.data)
            
            # Combine them
            cube = mean(alldata, allmasks)
        
            result = create_result(cube[0], variance=cube[1], exmap=cube[2])
        
            return {'qa': numina.qa.UNKNOWN, 'dark.fits': result}

        finally:
            for n in self.values['images']:
                n.close()

    