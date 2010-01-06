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

'''Recipe for the reduction of imaging mode observations.

Recipe to reduce observations obtained in imaging mode, considering different
possibilities depending on the size of the offsets between individual images.
In particular, the following strategies are considered: stare imaging, nodded
beamswitched imaging, and dithered imaging. 

A critical piece of information here is a table that clearly specifies which
images can be labeled as *science*, and which ones as *sky*. Note that some
images are used both as *science* and *sky* (when the size of the targets are
small compared to the offsets).

**Inputs:**

 * Science frames + [Sky Frames]
 * An indication of the observing strategy: **stare image**, **nodded
   beamswitched image**, or **dithered imaging**
 * A table relating each science image with its sky image(s) (TBD if it's in 
   the FITS header and/or in other format)
 * Offsets between them (Offsets must be integer)
 * Master Dark 
 * Bad pixel mask (BPM) 
 * Non-linearity correction polynomials 
 * Master flat (twilight/dome flats)
 * Master background (thermal background, only in K band)
 * Exposure Time (must be the same in all the frames)
 * Airmass for each frame
 * Detector model (gain, RN, lecture mode)
 * Average extinction in the filter
 * Astrometric calibration (TBD)

**Outputs:**

 * Image with three extensions: final image scaled to the individual exposure
   time, variance  and exposure time map OR number of images combined (TBD)

**Procedure:**

Images are corrected from dark, non-linearity and flat. Then, an iterative
process starts:

 * Sky is computed from each frame, using the list of sky images of each
   science frame. The objects are avoided using a mask (from the second
   iteration on).

 * The relative offsets are the nominal from the telescope. From the second
   iteration on, we refine them using objects of appropriate brightness (not
   too bright, not to faint).

 * We combine the sky-subtracted images, output is: a new image, a variance
   image and a exposure map/number of images used map.

 * An object mask is generated.

 * We recompute the sky map, using the object mask as an additional imput. From
   here we iterate (tipically 4 times).

 * Finally, the images are corrected from atmospheric extinction and flux
   calibrated.

 * A preliminary astrometric calibration can always be used (using the central
   coordinates of the pointing and the plate scale in the detector). A better
   calibration might be computed using available stars (TBD).

'''

__version__ = "$Revision$"

import os.path
import logging

import pyfits
import numpy

from numina.recipes import RecipeBase, RecipeResult
from numina.recipes import ParametersDescription, systemwide_parameters
#from numina.exceptions import RecipeError
from numina.image.processing import DarkCorrector, NonLinearityCorrector, FlatFieldCorrector
from numina.image.processing import generic_processing
from numina.image.combine import median
from emir.recipes import pipeline_parameters
import numina.qa as QA

_logger = logging.getLogger("emir.recipes")

_param_desc = ParametersDescription(inputs={'images': [],
                                            'master_bias': '',
                                            'master_dark': '',
                                            'master_flat': '',
                                            'master_bpm': ''},
                                    outputs={'bias': 'bias.fits'},
                                    optional={'linearity': (1.0, 0.00),
                                              },
                                    pipeline=pipeline_parameters(),
                                    systemwide=systemwide_parameters()
                                    )

def parameters_description():
    return _param_desc

class Result(RecipeResult):
    '''Result of the imaging mode recipe.'''
    def __init__(self, qa):
        super(Result, self).__init__(qa)


class Recipe(RecipeBase):
    '''Recipe to process data taken in imaging mode.
     
    '''
    def __init__(self, parameters):
        super(Recipe, self).__init__(parameters)
        
    def process(self):

        # dark correction
        # open the master dark
        dark_data = pyfits.getdata(self.parameters.inputs['master_dark'])    
        flat_data = pyfits.getdata(self.parameters.inputs['master_flat'])
        
        corrector1 = DarkCorrector(dark_data)
        corrector2 = NonLinearityCorrector(self.parameters.inputs['linearity'])
        corrector3 = FlatFieldCorrector(flat_data)
        
        generic_processing(self.parameters.inputs['images'], 
                           [corrector1, corrector2, corrector3], backup=True)
        
        del dark_data
        del flat_data    
        
        # Illumination seems to be necessary
        # ----------------------------------
        alldata = []
        
        for n in self.parameters.inputs['images']:
            f = pyfits.open(n, 'readonly', memmap=True)
            alldata.append(f[0].data)
            
        print alldata[0].shape
        
        allmasks = []
        #for n in ['apr21_0067DLFS-0.fits.mask'] * len(options.files):
        for n in self.parameters.inputs['images']:
            #f = pyfits.open(n, 'readonly', memmap=True)
            allmasks.append(numpy.zeros(alldata[0].shape))
        
        # Compute the median of all images in valid pixels
        scales = [numpy.median(data[mask == 0]) 
                  for data, mask in zip(alldata, allmasks)]
        
        illum_data = median(alldata, allmasks, scales=scales)
        print illum_data[0].shape
        print illum_data.mean()
        # Combining all the images
        
        
        # Data pre processed
        number_of_iterations = 4
        
        # ** 2 iter for bright objects + sextractor tunning
        # ** 4 iter for dim objects + sextractor tunning
        
        # ** QA after 1st iter
        # * Flux control
        # * Offset refinement

        # first iteration, without segmentation mask
        
        # Compute the initial sky subtracted images
        return Result(QA.UNKNOWN)
        current_iteration = 0
       
        for f in self.parameters.inputs['images']:
            # open file            
            # get the data
            newf = self.get_processed(f, 'DLFS-%d' % current_iteration)
            if os.path.lexists(newf) and not options.clobber:                
                _logger.info('File %s exists, skipping', newf)
                continue
            
            pf = self.get_processed(f, 'DLF')
            (file_data, file_header) = pyfits.getdata(pf, header=True)
            
            # Initial estimate of the sky background
            m = numpy.median(file_data)
            _logger.info('Sky value for image %s is %f', pf, m)
            file_data -= m
            
            _logger.info('Processing %s', newf)
            
        
        
        return Result()   
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
    from numina.user import main
    from numina.recipes import Parameters
    import json
    from numina.jsonserializer import to_json
     
    pv = {'inputs' :  {'images': ['apr21_0046.fits', 'apr21_0047.fits', 
                                  'apr21_0048.fits','apr21_0049.fits', 
                                  'apr21_0050.fits'],
                        'master_bias': 'mbias.fits',
                        'master_dark': 'Dark50.fits',
                        'linearity': (1e-3, 1e-2, 0.99, 0.00),
                        'master_flat': 'DummyFlat.fits',
                        'master_bpm': 'bpm.fits'
                        },
          'outputs' : {'bias': 'bias.fits'},
          'optional' : {},
          'pipeline' : {},
          'systemwide' : {'compute_qa': True}
    }
    
    p = Parameters(**pv)
    
    os.chdir('/home/sergio/IR/apr21')
    
    with open('config-d.txt', 'w+') as f:
        json.dump(p, f, default=to_json, encoding='utf-8', indent=2)
    
    main(['-d', '--run', 'direct_imaging', 'config-d.txt'])
