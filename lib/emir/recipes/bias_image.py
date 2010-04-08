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

import pyfits

import numina.recipes as nr
from emir.instrument.headers import EmirImage
from numina.array.combine import mean
import numina.qa as qa

_logger = logging.getLogger("emir.recipes")

class ParameterDescription(nr.ParameterDescription):
    def __init__(self):
        inputs={'images': []}
        optional={}
        super(ParameterDescription, self).__init__(inputs, optional)

class Result(nr.RecipeResult):
    '''Result of the recipe.'''
    def __init__(self, qa, hdulist):
        super(Result, self).__init__(qa)
        self.products['bias'] = hdulist

class Recipe(nr.RecipeBase):
    '''Recipe to process data taken in Bias image Mode.
    
    Here starts the long description...
    It continues several lines
    
    '''
    def __init__(self):
        super(Recipe, self).__init__()
        self.images = []
        
    def setup(self, param):
        super(Recipe, self).setup(param)
        self.images = self.inputs['images']
        
    def process(self):
        
        # Sanity check, check: all images belong to the same detector mode
        
        # Open all zero images
        alldata = []
        for n in self.images:
            f = pyfits.open(n, 'readonly', memmap=True)
            alldata.append(f[0].data)
        
        allmasks = []
        
        # Combine them
        cube = mean(alldata, allmasks)
        
        creator = EmirImage()
        hdulist = creator.create(cube[0], extensions=[('VARIANCE', cube[1], None)])
        
        if True:
            cqa = qa.GOOD
        else:
            cqa = qa.UNKNOWN
        
        return Result(cqa, hdulist)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
    from numina.user import main
    from numina.recipes import Parameters
    from numina.jsonserializer import to_json
    import simplejson as json
    import os
    
    pv = {'inputs' : {'images': ['apr21_0046.fits', 'apr21_0047.fits', 'apr21_0048.fits',
                     'apr21_0049.fits', 'apr21_0050.fits']},
         'optional' : {},
    }
    
    p = Parameters(**pv)
    
    os.chdir('/home/spr/Datos/emir/apr21')
    
    f = open('config.txt', 'w+')
    try:
        json.dump(p, f, default=to_json, encoding='utf-8',indent=2)
    finally:
        f.close()
    
    # main(['--list'])
    main(['-d', '--run', 'bias_image','config.txt'])

