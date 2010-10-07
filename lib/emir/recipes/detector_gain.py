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

import numpy
import scipy.stats

#from numina.array.combine import zerocombine
#from numina.image import DiskImage
import numina.qa
from numina.recipes import RecipeBase
#from emir.dataproducts import create_result
from emir.recipes import EmirRecipeMixin
from numina.recipes.registry import ProxyPath
from numina.recipes.registry import Schema
from numina.image import DiskImage

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
        # Complete detector
        CHANNELS1 = [(slice(0, 2048, None), slice(0, 2048, None))]
        # Quadrants 
        CHANNELS2 = [(slice(0, 1024, None), slice(0, 1024, None)), 
                     (slice(0, 1024, None), slice(1024, 2048, None)),
                     (slice(1024, 2048, None), slice(0, 1024, None)),
                     (slice(1024, 2048, None), slice(1024, 2048, None)),
        ]
        
        
        channels = CHANNELS2
        ramps = self.parameters['ramps']
        final = numpy.zeros((len(ramps), len(channels)))
        final2 = numpy.zeros((len(ramps), len(channels)))
        for ir, ramp in enumerate(ramps):
            vall = 0
            ffinal = final[ir]
            ffinal2 = final2[ir]
            store = numpy.zeros((len(ramp), len(channels)))
            store2 = numpy.zeros_like(store)
            for i, fname in enumerate(ramp):
                dname = DiskImage(fname)
                # Load file to data structure
                sstore = store[i]
                sstore2 = store2[i]
                for j, channel in enumerate(channels):
                    #print fname, channel
                    dname.open(mode='readonly')
                    sstore[j] = dname.data[channel].mean()
                    sstore2[j] = dname.data[channel].var(ddof=1)
                    dname.close()
                vall += 1
            for j, _ in enumerate(channels):
                ig, ron,_,_,_ = scipy.stats.linregress(store[:,j], store2[:,j])

                ffinal[j] = 1.0 / ig
                ffinal2[j] = ron
        gch_mean = final.mean(axis=0)
        gch_var = final.var(axis=0, ddof=1)
        print final
        rch_mean = final2.mean(axis=0)
        rch_var = final2.var(axis=0, ddof=1)
        
        return {'qa': numina.qa.UNKNOWN, 'gain': {'mean': list(gch_mean.flat), 
                                                  'var': list(gch_var.flat)},
                                          'ron': {'mean': list(rch_mean.flat), 
                                                  'var': list(rch_var.flat)}}
        
    
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
                                             ['r0000.fits', 'r0001.fits','r0002.fits', 'r0003.fits',
                                               'r0004.fits', 'r0005.fits', 'r0006.fits', 'r0007.fits'],
                                            ['r0008.fits', 'r0009.fits','r0010.fits', 'r0011.fits',
                                               'r0012.fits', 'r0013.fits', 'r0014.fits', 'r0015.fits'],
                                               
                                             ]
                            },
                       },                     
    }
        
    os.chdir('/home/spr/Datos/emir/test5')
    
    ff = open('config.txt', 'w+')
    try:
        json.dump(pv, ff, default=to_json, encoding='utf-8', indent=2)
    finally:
        ff.close()

    main(['-d', '--run', 'config.txt'])
