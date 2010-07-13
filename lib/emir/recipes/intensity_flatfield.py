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


import logging

import numpy

import numina.recipes as nr
from numina.image import DiskImage
from numina.image.flow import SerialFlow
from numina.image.processing import DarkCorrector, NonLinearityCorrector
from numina.array.combine import flatcombine
from numina.worker import para_map
from emir.instrument.headers import EmirImageCreator
import numina.qa as QA

_logger = logging.getLogger("emir.recipes")

class Result(nr.RecipeResult):
    '''Result of the intensity flat-field mode recipe.'''
    def __init__(self, qa, flat):
        super(Result, self).__init__(qa)
        self.products['flat'] = flat


class Recipe(nr.RecipeBase):
    '''Recipe to process data taken in intensity flat-field mode.
     
    '''
    required_parameters = [
        'images',
        'master_bias',
        'master_dark',
        'master_bpm',
        'nonlinearity',
        'nthreads',                         
    ]
    
    
    def __init__(self, values):
        super(Recipe, self).__init__(values)

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
        nthreads = self.values['nthreads']
        
        simages = [DiskImage(filename=i) for i in self.values['images']]
        smasks =  [self.values['master_bpm']] * len(simages)
                       
        # Initialize processing nodes, step 1
        try:
            dark_data = self.values['master_dark'].open(mode='readonly',
                                                        memmap=True)
            sss = SerialFlow([
                          DarkCorrector(dark_data),
                          NonLinearityCorrector(self.values['nonlinearity']),
                          ]
            )
        
            _logger.info('Basic processing')    
            para_map(lambda x : Recipe.f_basic_processing(x, sss), zip(simages, smasks), 
                     nthreads=nthreads)
        finally:
            self.values['master_dark'].close()
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
            map(lambda x: x.open(mode='readonly'), simages)
            map(lambda x: x.open(mode='readonly'), smasks)
            _logger.info("Combining the images without offsets")
            data = [img.data for img in simages]
            masks = [img.data for img in simages]
            _logger.info("Data combined")
            illum_data = flatcombine(data, masks, scales, blank=1)
        finally:
            map(lambda x: x.close(), simages)
            map(lambda x: x.close(), smasks)
    
        fc = EmirImageCreator()
                
        _logger.info("Final image created")
        extensions = [('VARIANCE', illum_data[1], None), 
                      ('NUMBER', illum_data[2], None)]
        illum = fc.create(illum_data[0], None, extensions)
        
        return Result(QA.UNKNOWN, illum)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
    from numina.user import main
    import simplejson as json
    import os
    from numina.jsonserializer import to_json

    pv = {'images': ['apr21_0046.fits',
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
                        'master_bias': DiskImage('mbias.fits'),
                        'master_dark': DiskImage('Dark50.fits'),
                        'linearity': [1e-3, 1e-2, 0.99, 0.00],
                        'master_bpm': DiskImage('bpm.fits')
    }
    
    os.chdir('/home/spr/Datos/emir/apr21')
    
    f = open('config-iff.txt', 'w+')
    try:
        json.dump(pv, f, default=to_json, encoding='utf-8')
    finally:
        f.close()
        
    main(['-d', '--run', 'intensity_flatfield', 'config-iff.txt'])
