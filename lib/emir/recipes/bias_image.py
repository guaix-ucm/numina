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

'''Bias image recipe and associates.

Recipe to process bias images. Bias images only appear in Simple Readout mode.

**Observing modes:**

 * Bias Image (3.1)   

**Inputs:**

 * A list of bias images
 * A model of the detector (gain, RN)

**Outputs:**

 * A combined bias frame, with variance extension and quality flag. 

**Procedure:**

The list of images can be readly processed by combining them with a typical
sigma-clipping algorithm.

'''

import logging

from numina.recipes import RecipeResult, RecipeBase
from emir.dataproducts import create_result
from numina.array.combine import mean
import numina.qa

_logger = logging.getLogger("emir.recipes")

class RecipeInput(object):
    def __init__(self):
        pass

class Result(RecipeResult):
    '''Result of the recipe.'''
    def __init__(self, qa, result):
        super(Result, self).__init__(qa)
        self.products['bias'] = result

class Recipe(RecipeBase):
    '''Recipe to process data taken in Bias image Mode.
    
    Here starts the long description...
    It continues several lines
    
    '''
    
    required_parameters = [
        'nthreads',
        'images',
    ]
    
    def __init__(self, values):
        super(Recipe, self).__init__(values)
        
    def run(self):
        
        # Sanity check, check: all images belong to the same detector mode
        
        # Open all zero images
        alldata = []
        allmasks = []
        try:
            for n in self.values['images']:
                n.open(mode='readonly', memmap=True)
                alldata.append(n.data)
            
            # Combine them
            cube = mean(alldata, allmasks)
        
            result = create_result(cube[0], 
                                   variance=cube[1], 
                                   exmap=cube[2].astype('int16'))
            if True:
                cqa = numina.qa.GOOD
            else:
                cqa = numina.qa.UNKNOWN
        
            return Result(cqa, result)
        finally:
            for n in self.values['images']:
                n.close()

if __name__ == '__main__':
    import os
    
    import simplejson as json
    
    from numina.image import Image
    from numina.user import main
    from numina.jsonserializer import to_json

    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
    
    pv = {'images': [Image('apr21_0046.fits'), 
                     Image('apr21_0047.fits'), 
                     Image('apr21_0048.fits'),
                     Image('apr21_0049.fits'), 
                     Image('apr21_0050.fits')
                     ]
    }
    
    
    os.chdir('/home/spr/Datos/emir/apr21')
    
    ff = open('config.txt', 'w+')
    try:
        json.dump(pv, ff, default=to_json, encoding='utf-8', indent=2)
    finally:
        ff.close()
    
    # main(['--list'])
    main(['-d', '--run', 'bias_image','config.txt'])

