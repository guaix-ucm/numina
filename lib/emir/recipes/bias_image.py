#
# Copyright 2008-2009 Sergio Pascual, Nicolas Cardiel
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

'''Bias image recipe and associates.

Recipe to process bias images. Bias images only appear in Simple Readout mode
(check this).

**Inputs:**

 * A list of bias images
 * A model of the detector (gain, RN)

**Outputs:**

 * A combined bias frame, with variance extension and quality flag. 

**Procedure:**

The list of images can be readly processed by combining them with a typical
sigma-clipping algorithm.

'''

from __future__ import with_statement

__version__ = "$Revision$"


import logging

import pyfits

from emir.recipes import pipeline_parameters
from emir.simulation.headers import default_fits_headers
from numina.recipes import RecipeBase, RecipeResult, ParametersDescription, systemwide_parameters
from numina.image.storage import FITSCreator
from numina.image.combine import mean
import numina.qa as qa

# Classes are new style
__metaclass__ = type

_logger = logging.getLogger("emir.recipes")

_param_desc = ParametersDescription(inputs={'images': []}, 
                                    outputs={'bias': 'bias.fits'}, 
                                    optional={}, 
                                    pipeline=pipeline_parameters(), 
                                    systemwide=systemwide_parameters()
                                    )

def parameters_description():
    return _param_desc

class Result(RecipeResult):
    '''Result of the recipe.'''
    def __init__(self, hdulist, qa):
        super(Result, self).__init__(qa)
        self.bias = hdulist
        
    def store(self):
        '''Description of store.
        
        :rtype: None'''
        pass
        #return self.hdulist.writeto(self.file, clobber=True, output_verify='ignore')


class Recipe(RecipeBase):
    '''Recipe to process data taken in Bias image Mode.
    
    Here starts the long description...
    It continues several lines
    
    '''
    #usage = 'usage: %prog [options] recipe [recipe-options]'
    #cmdoptions = OptionParser(usage=usage)
    #cmdoptions.add_option('--docs', action="store_true", dest="docs", 
    #                      default=False, help="prints documentation")
    def __init__(self, parameters):
        super(Recipe, self).__init__(parameters)
        
    def process(self):
        
        # Sanity check, check: all images belong to the same detector mode
        
        # Open all zero images
        alldata = []
        for n in self.parameters.inputs['images']:
            f = pyfits.open(n, 'readonly', memmap=True)
            alldata.append(f[0].data)
        
        allmasks = []
        
        # Combine them
        cube = mean(alldata, allmasks)
        
        creator = FITSCreator(default_fits_headers)
        hdulist = creator.create(cube[0], extensions=[('VARIANCE', cube[1], None)])
        
        if self.parameters.systemwide['compute_qa']:
            cqa = qa.GOOD
        else:
            cqa = qa.UNKNOWN
        
        return Result(hdulist, cqa)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
    from numina.user import main
    from numina.recipes import Parameters
    from numina.jsonserializer import to_json
    import json
    import os
    
    pv = {'inputs' : {'images': ['apr21_0046.fits', 'apr21_0047.fits', 'apr21_0048.fits',
                     'apr21_0049.fits', 'apr21_0050.fits']},
         'outputs' : {'bias': 'bias.fits'},
         'optional' : {},
         'pipeline' : {},
         'systemwide' : {'compute_qa': True}
    }
    
    p = Parameters(**pv)
    
    os.chdir('/home/inferis/spr/IR/apr21')
    
    with open('config.txt', 'w+') as f:
        json.dump(p, f, default=to_json, encoding='utf-8',indent=2)
    
    # main(['--list'])
    main(['-d', '--run', 'bias_image','config.txt'])

