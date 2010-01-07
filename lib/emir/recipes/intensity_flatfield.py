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

**Inputs:**

 * TBD

**Outputs:**

 * TBD

**Procedure:**

 * TBD

'''

__version__ = "$Revision$"

import logging
import warnings

import pyfits
import numpy

from numina.recipes import RecipeBase, RecipeResult
from numina.recipes import ParametersDescription, systemwide_parameters
#from numina.exceptions import RecipeError
from numina.image.processing import DarkCorrector, NonLinearityCorrector
from numina.image.processing import generic_processing
from numina.image.combine import median, mean
from emir.recipes import pipeline_parameters
from emir.instrument.headers import EmirImage
import numina.qa as QA

_logger = logging.getLogger("emir.recipes")

_param_desc = ParametersDescription(inputs={'images': [],
                                            'masks': [],
                                            'master_bias': '',
                                            'master_dark': '',
                                            'master_bpm': ''},
                                    outputs={'flat': 'flat.fits'},
                                    optional={'linearity': (1.0, 0.00),
                                              },
                                    pipeline=pipeline_parameters(),
                                    systemwide=systemwide_parameters()
                                    )

def parameters_description():
    return _param_desc

class Result(RecipeResult):
    '''Result of the intensity flat-field mode recipe.'''
    def __init__(self, qa, flat):
        super(Result, self).__init__(qa)
        self.flat = flat


class Recipe(RecipeBase):
    '''Recipe to process data taken in intensity flat-field mode.
     
    '''
    def __init__(self, parameters):
        super(Recipe, self).__init__(parameters)
        
    def process(self):

        # dark correction
        # open the master dark    
        dark_data = pyfits.getdata(self.parameters.inputs['master_dark'])    
        
        corrector1 = DarkCorrector(dark_data)
        corrector2 = NonLinearityCorrector(self.parameters.inputs['linearity'])
             
        generic_processing(self.parameters.inputs['images'], 
                           [corrector1, corrector2], backup=True)
          
        
        # Illumination seems to be necessary
        # ----------------------------------
        alldata = []
        
        for n in self.parameters.inputs['images']:
            f = pyfits.open(n, 'readonly', memmap=False)
            alldata.append(f['PRIMARY'].data)
        
        allmasks = []
        
        for n in self.parameters.inputs['masks']:
            f = pyfits.open(n, 'readonly', memmap=False)
            allmasks.append(f['PRIMARY'].data)
        
        
        # Compute the median of all images in valid pixels
        scales = [numpy.mean(data) 
                  for data, mask in zip(alldata, allmasks)]
        _logger.info("Scales computed")
        
        # Combining all the images
        illum_data = mean(alldata, masks=allmasks, scales=scales)
        _logger.info("Data combined")
        
        fc = EmirImage()
        
        illum = fc.create(illum_data)
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
    import numpy as np
     
    pv = {'inputs' :  {'images': ['apr21_0067.fits', 'apr21_0068.fits', 
                                  'apr21_0069.fits','apr21_0070.fits'],
                        'masks': ['apr21_0067.fits_mask', 'apr21_0068.fits_mask', 
                                  'apr21_0069.fits_mask','apr21_0070.fits_mask'],
                        'master_bias': 'mbias.fits',
                        'master_dark': 'Dark50.fits',
                        'linearity': [1e-3, 1e-2, 0.99, 0.00],
                        'master_bpm': 'bpm.fits'
                        },
          'outputs' : {},
          'optional' : {},
          'pipeline' : {},
          'systemwide' : {'compute_qa': True}
    }
    
    p = Parameters(**pv)
    
    os.chdir('/home/sergio/IR/apr21')
    
    #with open('config-iff.txt', 'w+') as f:
    #    json.dump(p, f, default=to_json, encoding='utf-8', indent=2)
    
    main(['-d', '--run', 'intensity_flatfield', 'config-iff.txt'])
