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

# $Id$

'''Intensity Flatfield Recipe.

Recipe to process intensity flat-fields. The flat-on and flat-off images are
combined (method?) separately and the subtracted to obtain a thermal subtracted
flat-field.

**Observing modes:**

 * Intensity Flat-Field

**Inputs:**

  * A list of lamp-on flats
  * A list of lamp-off flats
  * A master dark frame
  * A model of the detector. 

**Outputs:**

 * TBD

**Procedure:**

 * A combined thermal subtracted flat field, normalized to median 1, 
   with with variance extension and quality flag. 

'''

__version__ = "$Revision$"

import logging

import pyfits
import numpy as np

import numina.recipes as nr
from numina.image.processing import DarkCorrector, NonLinearityCorrector
from numina.image.processing import generic_processing
from numina.array.combine import mean
from emir.instrument.headers import EmirImage
import numina.qa as QA

_logger = logging.getLogger("emir.recipes")

class ParameterDescription(nr.ParameterDescription):
    def __init__(self):
        inputs={'images': [],
                'masks': [],
                'master_bias': '',
                'master_dark': '',
                'master_bpm': ''}
        optional={'linearity': (1.0, 0.00),}
        super(ParameterDescription, self).__init__(inputs, optional)


class Result(nr.RecipeResult):
    '''Result of the intensity flat-field mode recipe.'''
    def __init__(self, qa, flat):
        super(Result, self).__init__(qa)
        self.products['flat'] = flat


class Recipe(nr.RecipeBase):
    '''Recipe to process data taken in intensity flat-field mode.
     
    '''
    def __init__(self):
        super(Recipe, self).__init__()

    def initialize(self, param):
        super(Recipe, self).initialize(param)

    def process(self):

        # dark correction
        # open the master dark    
        dark_data = pyfits.getdata(self.inputs['master_dark'])    
        
        corrector1 = DarkCorrector(dark_data)
        corrector2 = NonLinearityCorrector(self.inputs['linearity'])
             
        generic_processing(self.inputs['images'],
                           [corrector1, corrector2], backup=True)
          
        
        # Illumination seems to be necessary
        # ----------------------------------
        alldata = []
        
        for n in self.inputs['images']:
            f = pyfits.open(n, 'readonly', memmap=True)
            _logger.debug('Loading image %s', n)
            d = f['PRIMARY'].data
            if np.any(np.isnan(d)):
                _logger.warning('Image %s has NaN values', n)
            else:
                alldata.append(d) 
        
        allmasks = []
        
        for n in self.inputs['masks']:
            f = pyfits.open(n, 'readonly', memmap=True)
            _logger.debug('Loading mask %s', n)
            d = f['PRIMARY'].data
            if np.any(np.isnan(d)):
                _logger.warning('Mask %s has NaN values', n)
            else:
                allmasks.append(d)
        
        # Compute the median of all images in valid pixels
        scales = [np.mean(data[mask == 0]) 
                  for data, mask in zip(alldata, allmasks)]
        _logger.info("Scales computed")
        
        # Combining all the images
        illum_data = mean(alldata, masks=allmasks, scales=scales)
        _logger.info("Data combined")

        fc = EmirImage()
        
        substitute_value = 1
        illum_data[illum_data == 0] = substitute_value
        
        illum = fc.create(illum_data[0])
        _logger.info("Final image created")
        
        return Result(QA.UNKNOWN, illum)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
    from numina.user import main
    from numina.recipes import Parameters
    import json
    import os
    from numina.jsonserializer import to_json

    pv = {'inputs' :  {'images': ['apr21_0046.fits',
                                  'apr21_0047.fits',
                                  'apr21_0048.fits',
                                  'apr21_0049.fits',
                                  'apr21_0051.fits',
                                  'apr21_0052.fits',
                                  'apr21_0053.fits',
                                  'apr21_0054.fits',
                                  'apr21_0055.fits',
                                  'apr21_0056.fits',
                                  'apr21_0057.fits',
                                  'apr21_0058.fits',
                                  'apr21_0059.fits',
                                  'apr21_0060.fits',
                                  'apr21_0061.fits',
                                  'apr21_0062.fits',
                                  'apr21_0063.fits',
                                  'apr21_0064.fits',
                                  'apr21_0065.fits',
                                  'apr21_0066.fits',
                                  'apr21_0067.fits',
                                  'apr21_0068.fits',
                                  'apr21_0069.fits',
                                  'apr21_0070.fits',
                                  'apr21_0071.fits',
                                  'apr21_0072.fits',
                                  'apr21_0073.fits',
                                  'apr21_0074.fits',
                                  'apr21_0075.fits',
                                  'apr21_0076.fits',
                                  'apr21_0077.fits',
                                  'apr21_0078.fits',
                                  'apr21_0079.fits',            
                                  'apr21_0080.fits',
                                  'apr21_0081.fits'],
                        'masks': ['apr21_0046.fits_mask',
                                  'apr21_0047.fits_mask',
                                  'apr21_0048.fits_mask',
                                  'apr21_0049.fits_mask',                                  
                                  'apr21_0051.fits_mask',          
                                  'apr21_0052.fits_mask',
                                  'apr21_0053.fits_mask',
                                  'apr21_0054.fits_mask',
                                  'apr21_0055.fits_mask',
                                  'apr21_0056.fits_mask',
                                  'apr21_0057.fits_mask',
                                  'apr21_0058.fits_mask',
                                  'apr21_0059.fits_mask',
                                  'apr21_0060.fits_mask',          
                                  'apr21_0061.fits_mask',
                                  'apr21_0062.fits_mask',
                                  'apr21_0063.fits_mask',
                                  'apr21_0064.fits_mask',
                                  'apr21_0065.fits_mask',
                                  'apr21_0066.fits_mask',
                                  'apr21_0067.fits_mask',
                                  'apr21_0068.fits_mask',
                                  'apr21_0069.fits_mask',
                                  'apr21_0070.fits_mask',
                                  'apr21_0071.fits_mask',
                                  'apr21_0072.fits_mask',
                                  'apr21_0073.fits_mask',
                                  'apr21_0074.fits_mask',
                                  'apr21_0075.fits_mask',
                                  'apr21_0076.fits_mask',
                                  'apr21_0077.fits_mask',
                                  'apr21_0078.fits_mask',
                                  'apr21_0079.fits_mask',
                                  'apr21_0080.fits_mask',
                                  'apr21_0081.fits_mask'],
                        'master_bias': 'mbias.fits',
                        'master_dark': 'Dark50.fits',
                        'linearity': [1e-3, 1e-2, 0.99, 0.00],
                        'master_bpm': 'bpm.fits'
                        },
                'optional' : {}
    }
    
    p = Parameters(**pv)
    
    os.chdir('/home/spr/Datos/emir/apr21')
    
    f = open('config-iff.txt', 'w+')
    try:
        json.dump(p, f, default=to_json, encoding='utf-8', indent=2)
    finally:
        f.close()
        
    main(['-d', '--run', 'intensity_flatfield', 'config-iff.txt'])
