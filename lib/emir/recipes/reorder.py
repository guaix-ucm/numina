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

'''Recipe for the reordering of frames.'''

import logging

import numpy
import os.path

import numina.qa
from numina.recipes import RecipeBase
#from emir.dataproducts import create_result
from emir.recipes import EmirRecipeMixin
from numina.recipes.registry import ProxyPath
from numina.recipes.registry import Schema
from numina.image import DiskImage
from emir.instrument.detector import CHANNELS, QUADRANTS
from emir.dataproducts import create_result

_logger = logging.getLogger("emir.recipes")

class Recipe(RecipeBase, EmirRecipeMixin):
    '''Reordering Recipe.
    
    Recipe to reorder images created by the detector.
    '''
    required_parameters = [
        Schema('images', ProxyPath('/observing_block/result/images'), 
               'A list of images created by the detector'),
    ]
    capabilities = ['detector_reorder']
    
    
    def __init__(self, param, runinfo):
        super(Recipe, self).__init__(param, runinfo)
        _logger.info('building')

    def region_full(self):
        return [(slice(0, 2048), slice(0, 2048))]
    
    def region_quadrant(self):
        return QUADRANTS
    
    def region_channel(self):
        return CHANNELS

    def region(self):
        fun = getattr(self, 'region_%s' %  self.parameters['region'])  
        return fun()
    
    def partition(self, pred, seq, maxval):
        final = [[] for i in range(maxval)]
        assert len(final) == maxval
        for i in seq:
            val = pred(i)
            final[val].append(i)
                
        return final
    
    def partition2(self, pred, seq):
        final = {}
        for i in seq:
            val = pred(i)
            if val in final:
                final[val].append(i)
            else:
                final[val] = [i]                
        return final
    

    def run(self):
        
        images = [DiskImage(os.path.join(self.runinfo['datadir'], path)) 
                  for path in self.parameters['images']]

        nquad = 4
        idx0, idx1, idx2, idx3 = self.partition(lambda x: (x / 128) % nquad, 
                                                range(4096), nquad)

        results = []

        for img in images:
            # Using numpy memmap instead of pyfits
            # Images are a in a format unrecognized by pyfits
            _logger.debug('processing %s', img.filename)
            f = numpy.memmap(img.filename, 
                                  dtype='uint16', mode='r', offset=36 * 80)            
            try:                                
                f.shape = (1024, 4096)
                rr = numpy.zeros((2048, 2048), dtype='int16')
                # Fixme: signs missing here
                rr[0:1024, 0:1024] = f[:,idx0]
                rr[1024:2048, 0:1024] = f[:,idx1] 
                rr[1024:2048, 1024:2048] = f[:,idx2]
                rr[0:1024, 1024:2048] = f[:,idx3]  

                basename = os.path.basename(img.filename)
                primary_headers = {'FILENAME': basename}
                result = create_result(rr, 
                                       headers=primary_headers)
                result.writeto(basename)
                
                #result.close()                
                results.append(DiskImage(os.path.abspath(basename)))
            finally:
                del f
                img.close()
        
        return {'qa': numina.qa.UNKNOWN, 'images': results}
                                                  
    
if __name__ == '__main__':
    import uuid
    import glob
    
    import simplejson as json
    
    from numina.user import main
    from numina.jsonserializer import to_json

    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
        
    pv = {'recipes': {'default': {'parameters': {
                                    'region': 'channel'
                                },
                     'run': {'repeat': 1,
                             },   
                        },                 
                      },                      
          'observing_block': {'instrument': 'emir',
                       'mode': 'detector_reorder',
                       'id': 1,
                       'result': {
                                  'images': [], 

                            },
                       },                     
    }
    os.chdir('/home/spr/Datos/emir/test6/data')
    
    pv['observing_block']['result']['images'] = glob.glob('*serie*.fits')
    #pv['observing_block']['result']['images'] = ['pozo_BG33_VR05_OFF11_serie.0054.fits']
    os.chdir('/home/spr/Datos/emir/test6')
    
    # Creating base directory for storing results
    uuidstr = str(uuid.uuid1()) 
    basedir = os.path.join(os.getcwd(), uuidstr)
    os.mkdir(basedir)
    
    ff = open('config.txt', 'w+')
    try:
        json.dump(pv, ff, default=to_json, encoding='utf-8', indent=2)
    finally:
        ff.close()

    main(['-d', '--basedir', basedir, '--datadir', '/home/spr/Datos/emir/test6/data',
          '--run', 'config.txt'])
