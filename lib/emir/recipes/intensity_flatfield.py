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

'''Intensity Flatfield Recipe.'''

import logging

import numpy

from numina.recipes import RecipeBase
from numina.image import DiskImage
from numina.image.flow import SerialFlow
from numina.image.processing import DarkCorrector, NonLinearityCorrector
from numina.array.combine import flatcombine
from numina.worker import para_map
from numina.recipes.registry import ProxyPath, ProxyQuery
from numina.recipes.registry import Schema
import numina.qa as QA
from emir.dataproducts import create_result
from emir.recipes import EmirRecipeMixin

_logger = logging.getLogger("emir.recipes")

class Recipe(RecipeBase, EmirRecipeMixin):
    '''Recipe to process data taken in intensity flat-field mode.
        
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
    
    required_parameters = [
        Schema('images', ProxyPath('/observing_block/result/images'), 'A list of paths to images'),        
        Schema('master_bias', ProxyQuery(), 'Master bias image'),
        Schema('master_dark', ProxyQuery(), 'Master dark image'),
        Schema('master_bpm', ProxyQuery(), 'Master bad pixel mask'),
        Schema('nonlinearity', ProxyQuery(dummy=[1.0, 0.0]), 'Polynomial for non-linearity correction'),        
        Schema('output_filename', 'result.fits', 'Name of the output image')
    ]
    
    capabilities = ['intensity_flatfield']
    
    def __init__(self, param, runinfo):
        super(Recipe, self).__init__(param, runinfo)

    @staticmethod
    def f_basic_processing(od, flow):
        img = od[0]
        img.open(mode='update')
        try:
            img = flow(img)
            return od        
        finally:
            img.close(output_verify='fix')
            
    @staticmethod
    def f_flow4(od):        
        try:
            od[0].open(mode='readonly')
            od[1].open(mode='readonly')        
            
            d = od[0].data
            m = od[1].data
            value = numpy.median(d[m == 0])
            _logger.debug('median value of %s is %f', od[0], value)
            return od + (value,)    
        finally:
            od[1].close()
            od[0].close()            

    def run(self):
        
        nthreads = self.runinfo['nthreads']
        primary_headers = {'FILENAME': self.parameters['output_filename']}
                
        simages = [DiskImage(filename=i) for i in self.parameters['images']]
        smasks =  [DiskImage(self.parameters['master_bpm'])] * len(simages)
        
        dark_image = DiskImage(self.parameters['master_dark'])
        # Initialize processing nodes, step 1
        try:
            dark_data = dark_image.open(mode='readonly', memmap=True)
            sss = SerialFlow([
                          DarkCorrector(dark_data),
                          NonLinearityCorrector(self.parameters['nonlinearity']),
                          ]
            )
        
            _logger.info('Basic processing')    
            para_map(lambda x : Recipe.f_basic_processing(x, sss), zip(simages, smasks), 
                     nthreads=nthreads)
        finally:
            dark_image.close()
        # Illumination seems to be necessary
        # ----------------------------------
        
        _logger.info('Computing scale factors')
            
#        simages, smasks, scales = para_map(self.f_flow4, 
        intermediate = para_map(self.f_flow4,
                                           zip(simages, smasks), 
                                           nthreads=nthreads)
        simages, smasks, scales = zip(*intermediate)
            # Operation to create an intermediate sky flat
            
        try:
            map(lambda x: x.open(mode='readonly', memmap=True), simages)
            map(lambda x: x.open(mode='readonly', memmap=True), smasks)
            _logger.info("Combining the images without offsets")
            data = [img.data for img in simages]
            masks = [img.data for img in simages]
            _logger.info("Data combined")
            illum_data = flatcombine(data, masks, scales=scales, blank=1)
        finally:
            map(lambda x: x.close(), simages)
            map(lambda x: x.close(), smasks)
    
        _logger.info("Final image created")
        illum = create_result(illum_data[0], headers=primary_headers, 
                                   variance=illum_data[1], 
                                   exmap=illum_data[2])
        
        return {'qa': QA.UNKNOWN, 'illumination_image': illum}
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
    from numina.user import main
    import simplejson as json
    import os
    from numina.jsonserializer import to_json

    pv = {'recipe': {'parameters': 
                        {'images': ['apr21_0046.fits',
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
                                'master_bias': 'mbias.fits',
                                'master_dark': 'Dark50.fits',
                                'linearity': [1e-3, 1e-2, 0.99, 0.00],
                                'master_bpm': 'bpm.fits',
                                },
                        'run': {'mode': 'intensity_flatfield',
                                'instrument': 'emir'}
                        },
    }
    
    os.chdir('/home/spr/Datos/emir/apr21')
    
    f = open('config-iff.txt', 'w+')
    try:
        json.dump(pv, f, default=to_json, encoding='utf-8')
    finally:
        f.close()
        
    main(['-d', '--run', 'config-iff.txt'])
