#
# Copyright 2010 Sergio Pascual
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

'''Recipe for the reduction of gain calibration frames.'''

import logging

#from numina.array.combine import zerocombine
#from numina.image import DiskImage
import numina.qa
from numina.recipes import RecipeBase
#from emir.dataproducts import create_result
from emir.recipes import EmirRecipeMixin
from numina.recipes.registry import ProxyPath
from numina.recipes.registry import Schema

_logger = logging.getLogger("emir.recipes")

class Recipe(RecipeBase, EmirRecipeMixin):
    '''Detector Gain Recipe.
    
    Recipe to calibrate the detector gain.
    '''
    required_parameters = [
        Schema('resets', ProxyPath('/observing_block/result/reset'), 
               'A list of paths to reset images'),
        Schema('ramps', ProxyPath('/observing_block/result/ramps'), 
               'A list of ramps'),
    ]
    capabilities = ['detector_gain']
    
    
    def __init__(self, param, runinfo):
        super(Recipe, self).__init__(param, runinfo)

    def run(self):        
        return {'qa': numina.qa.UNKNOWN}
    
if __name__ == '__main__':
    import os
    
    import simplejson as json
    
    from numina.user import main
    from numina.jsonserializer import to_json

    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
        
    pv = {'recipe': {'parameters': {
                                'output_filename': 'perryr.fits',
                                'combine': 'average',
                                },
                     'run': {'repeat': 1,
                             'instrument': 'emir',
                             'mode': 'bias_image',
                             },   
                        },
          'observing_block': {'instrument': 'emir',
                       'mode': 'detector_gain',
                       'id': 1,
                       'result': {
                                  'resets': ['apr21_0046.fits', 
                                             'apr21_0047.fits', 
                                             'apr21_0048.fits',
                                             'apr21_0049.fits', 
                                             'apr21_0050.fits'
                                             ],
                                   'ramps': [
                                             ['ramp1'],
                                             ['ramp2'],
                                             ['ramp3']
                                             ]
                            },
                       },                     
    }
        
    os.chdir('/home/spr/Datos/emir/apr21')
    
    ff = open('config.txt', 'w+')
    try:
        json.dump(pv, ff, default=to_json, encoding='utf-8', indent=2)
    finally:
        ff.close()

    main(['-d', '--run', 'config.txt'])