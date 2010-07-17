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

'''Bias image recipe and associates.'''

import logging

from numina.array.combine import mean
import numina.qa
from numina.recipes import RecipeBase
from emir.dataproducts import create_result
from emir.recipes import EmirRecipeMixin

_logger = logging.getLogger("emir.recipes")

class Recipe(RecipeBase, EmirRecipeMixin):
    '''Recipe to process data taken in Bias image Mode.

    Bias images only appear in Simple Readout mode.

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
    
    required_parameters = [
        'nthreads',
        'images',
    ]

    capabilities = ['bias_image']
    
    def __init__(self, values):
        super(Recipe, self).__init__(values)
        
    def run(self):
        
        OUTPUT = 'bias.fits'
        primary_headers = {'FILENAME': OUTPUT}
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
        
            result = create_result(cube[0], headers=primary_headers,
                                   variance=cube[1], 
                                   exmap=cube[2].astype('int16'))
        
            return {'qa': numina.qa.UNKNOWN, 'bias_image': result}
        finally:
            for n in self.values['images']:
                n.close()

if __name__ == '__main__':
    import os
    
    import simplejson as json
    
    from numina.image import DiskImage
    from numina.user import main
    from numina.jsonserializer import to_json

    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
    
    pv = {'observing_mode': 'bias_image',
          'images': [DiskImage('apr21_0046.fits'), 
                     DiskImage('apr21_0047.fits'), 
                     DiskImage('apr21_0048.fits'),
                     DiskImage('apr21_0049.fits'), 
                     DiskImage('apr21_0050.fits')
                     ],
    }
    
    
    os.chdir('/home/spr/Datos/emir/apr21')
    
    ff = open('config.txt', 'w+')
    try:
        json.dump(pv, ff, default=to_json, encoding='utf-8', indent=2)
    finally:
        ff.close()

    main(['-d', '--run', 'config.txt'])

