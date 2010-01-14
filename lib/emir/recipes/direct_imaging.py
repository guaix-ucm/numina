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

 * We recompute the sky map, using the object mask as an additional input. From
   here we iterate (typically 4 times).

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
import numpy as np

from numina.recipes import RecipeBase, RecipeResult
from numina.recipes import ParametersDescription, systemwide_parameters
#from numina.exceptions import RecipeError
from numina.image.processing import DarkCorrector, NonLinearityCorrector, FlatFieldCorrector
from numina.image.processing import generic_processing
from numina.image.combine import median
from emir.recipes import pipeline_parameters
from emir.instrument.headers import EmirImage
import numina.qa as QA

_logger = logging.getLogger("emir.recipes")

_param_desc = ParametersDescription(inputs={'images': [],
                                            'masks': [],
                                            'offsets': [],
                                            'master_bias': '',
                                            'master_dark': '',
                                            'master_flat': '',
                                            'master_bpm': ''},
                                    outputs={'result': 'result.fits'},
                                    optional={'linearity': [1.0, 0.0],
                                              },
                                    pipeline=pipeline_parameters(),
                                    systemwide=systemwide_parameters()
                                    )

def parameters_description():
    return _param_desc

def combine_shape(shapes, offsets):
    # Computing final image size and new offsets
    sharr = np.asarray(shapes)
    
    offarr = np.asarray(offsets)        
    ucorners = offarr + sharr
    ref = offarr.min(axis=0)        
    finalshape = ucorners.max(axis=0) - ref 
    offsetsp = offarr - ref
    return (finalshape, offsetsp)

class Result(RecipeResult):
    '''Result of the imaging mode recipe.'''
    def __init__(self, qa, result):
        self.result = result
        super(Result, self).__init__(qa)
        


class Recipe(RecipeBase):
    '''Recipe to process data taken in imaging mode.
     
    '''
    def __init__(self, parameters):
        super(Recipe, self).__init__(parameters)
        
    def process(self):

        images = self.parameters.inputs['images'].keys()
        images.sort()

        # dark correction
        # open the master dark
        dark_data = pyfits.getdata(self.parameters.inputs['master_dark'])    
        flat_data = pyfits.getdata(self.parameters.inputs['master_flat'])
        
        corrector1 = DarkCorrector(dark_data)
        corrector2 = NonLinearityCorrector(self.parameters.inputs['linearity'])
        corrector3 = FlatFieldCorrector(flat_data)
        
        generic_processing(images, [corrector1, corrector2, corrector3], backup=True)
        
        del dark_data
        del flat_data    
        
        # Preiteration
        
        # Getting the offsets
        offsets = [self.parameters.inputs['images'][k][1] for k in images]
        masks = [self.parameters.inputs['images'][k][0] for k in images]
        
        # Getting the shapes of all the images
        allshapes = []
        
        # All images have the same shape
        for file in images:
            hdulist = pyfits.open(file)
            try:
                allshapes.append(hdulist['primary'].data.shape)
            finally:
                hdulist.close()
                
        # Getting the shapes of all the masks
        maskshapes = []
        
        # All images have the same shape
        for file in masks:
            hdulist = pyfits.open(file)
            try:
                maskshapes.append(hdulist['primary'].data.shape)
            finally:
                hdulist.close()
                
        # masksshapes and allshapes must be equal
        
        # Computing the shape of the fional image
        finalshape, offsetsp = combine_shape(allshapes, offsets)
        _logger.info("Shape of the final image %s", finalshape)
        
        # Resize images, to final size
        for file, shape, o in zip(images, allshapes, offsetsp):
            hdulist = pyfits.open(file)
            try:
                p = hdulist['primary']
                newfile = self.rescaled_image(file)
                newdata = np.zeros(finalshape, dtype=p.data.dtype)
                
                region = (slice(o[0], o[0] + shape[0]), slice(o[1], o[1] + shape[1]))
                newdata[region] = p.data
                pyfits.writeto(newfile, newdata, p.header, output_verify='silentfix', clobber=True)
                _logger.info('Resized image %s into image %s, new shape %s', file, newfile, finalshape)
            finally:
                hdulist.close()
                
        # Resize masks to final size
        for file, shape, o in zip(masks, allshapes, offsetsp):
            hdulist = pyfits.open(file)
            try:
                p = hdulist['primary']
                newfile = self.rescaled_image(file)
                newdata = np.zeros(finalshape, dtype=p.data.dtype)
                
                region = (slice(o[0], o[0] + shape[0]), slice(o[1], o[1] + shape[1]))
                newdata[region] = p.data
                pyfits.writeto(newfile, newdata, p.header, output_verify='silentfix', clobber=True)
                _logger.info('Resized mask %s into mask %s, new shape %s', file, newfile, finalshape)
            finally:
                hdulist.close()
                
        # Compute sky from the image (median)
        # Only for images that are Science images
        
        sky_backgrounds = []
        
        for file in images:
            # The sky images corresponding to file
            sky_images = self.parameters.inputs['images'][file][2]
            r_sky_images = [self.rescaled_image(i) for i in sky_images]
            r_sky_masks = [self.rescaled_image(self.parameters.inputs['images'][i][0]) 
                         for i in sky_images]
            _logger.info('Sky images for %s are %s', file, self.parameters.inputs['images'][file][2])
            
            # Combine the sky images with masks
            
            skyback = median(r_sky_images, r_sky_masks)
            median_sky = np.median(skyback[0][skyback[2] != 0])
            _logger.info('Median sky value is %d', median_sky)
            sky_backgrounds.append(median_sky)
        
        # Combine
        r_images = [self.rescaled_image(i) for i in images]
        r_masks = [self.rescaled_image(i) for i in masks]
        final_data = median(r_images, r_masks, zeros=sky_backgrounds)
        _logger.info('Combined images')
        # Generate object mask (using sextractor?)
        
        # Iterate 4 times
        
        fc = EmirImage()
        
        final = fc.create(final_data[0])
        _logger.info("Final image created")
        
        return Result(QA.UNKNOWN, final)
    
    def rescaled_image(self, name):
        return 'r_%s' % name

    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
    from numina.user import main
    from numina.recipes import Parameters
    import json
    from numina.jsonserializer import to_json
     
    pv = {'inputs' :  { 'images':  
                       {'apr21_0046.fits': ('apr21_0046.fits_mask', (0, 0), ['apr21_0046.fits']),
                        'apr21_0047.fits': ('apr21_0047.fits_mask', (0, 0), ['apr21_0047.fits']),
                        'apr21_0048.fits': ('apr21_0048.fits_mask', (0, 0), ['apr21_0048.fits']),
                        'apr21_0049.fits': ('apr21_0049.fits_mask', (21, -23), ['apr21_0049.fits']),
                        'apr21_0051.fits': ('apr21_0051.fits_mask', (21, -23), ['apr21_0051.fits']),
                        'apr21_0052.fits': ('apr21_0052.fits_mask', (-15, -35), ['apr21_0052.fits']),
                        'apr21_0053.fits': ('apr21_0053.fits_mask', (-15, -35), ['apr21_0053.fits']),
                        'apr21_0054.fits': ('apr21_0054.fits_mask', (-15, -35), ['apr21_0054.fits']),
                        'apr21_0055.fits': ('apr21_0055.fits_mask', (24, 12), ['apr21_0055.fits']),
                        'apr21_0056.fits': ('apr21_0056.fits_mask', (24, 12), ['apr21_0056.fits']),
                        'apr21_0057.fits': ('apr21_0057.fits_mask', (24, 12), ['apr21_0057.fits']),
                        'apr21_0058.fits': ('apr21_0058.fits_mask', (-27, 18), ['apr21_0058.fits']),
                        'apr21_0059.fits': ('apr21_0059.fits_mask', (-27, 18), ['apr21_0059.fits']),
                        'apr21_0060.fits': ('apr21_0060.fits_mask', (-27, 18), ['apr21_0060.fits']),
                        'apr21_0061.fits': ('apr21_0061.fits_mask', (-38, -16), ['apr21_0061.fits']),
                        'apr21_0062.fits': ('apr21_0062.fits_mask', (-38, -16), ['apr21_0062.fits']),
                        'apr21_0063.fits': ('apr21_0063.fits_mask', (-38, -17), ['apr21_0063.fits']),
                        'apr21_0064.fits': ('apr21_0064.fits_mask', (5, 27), ['apr21_0064.fits']),
                        'apr21_0065.fits': ('apr21_0065.fits_mask', (5, 27), ['apr21_0065.fits']),
                        'apr21_0066.fits': ('apr21_0066.fits_mask', (5, 27), ['apr21_0066.fits']),
                        'apr21_0067.fits': ('apr21_0067.fits_mask', (32, -13), ['apr21_0067.fits']),
                        'apr21_0068.fits': ('apr21_0068.fits_mask', (33, -13), ['apr21_0068.fits']),
                        'apr21_0069.fits': ('apr21_0069.fits_mask', (32, -13), ['apr21_0069.fits']),
                        'apr21_0070.fits': ('apr21_0070.fits_mask', (-52, 7), ['apr21_0070.fits']),
                        'apr21_0071.fits': ('apr21_0071.fits_mask', (-52, 8), ['apr21_0071.fits']),
                        'apr21_0072.fits': ('apr21_0072.fits_mask', (-52, 8), ['apr21_0072.fits']),
                        'apr21_0073.fits': ('apr21_0073.fits_mask', (-3, -49), ['apr21_0073.fits']),
                        'apr21_0074.fits': ('apr21_0074.fits_mask', (-3, -49), ['apr21_0074.fits']),
                        'apr21_0075.fits': ('apr21_0075.fits_mask', (-3, -49), ['apr21_0075.fits']),
                        'apr21_0076.fits': ('apr21_0076.fits_mask', (-49, -33), ['apr21_0076.fits']),
                        'apr21_0077.fits': ('apr21_0077.fits_mask', (-49, -32), ['apr21_0077.fits']),
                        'apr21_0078.fits': ('apr21_0078.fits_mask', (-49, -32), ['apr21_0078.fits']),
                        'apr21_0079.fits': ('apr21_0079.fits_mask', (-15, 36), ['apr21_0079.fits']),
                        'apr21_0080.fits': ('apr21_0080.fits_mask', (-16, 36), ['apr21_0080.fits']),
                        'apr21_0081.fits': ('apr21_0081.fits_mask', (-16, 36), ['apr21_0081.fits'])
                        },   
                        'master_bias': 'mbias.fits',
                        'master_dark': 'Dark50.fits',
                        'linearity': [1e-3, 1e-2, 0.99, 0.00],
                        'master_flat': 'flat.fits',
                        'master_bpm': 'bpm.fits'
                        },
          'outputs' : {},#'result': 'result.fits'},
          'optional' : {},
          'pipeline' : {},
          'systemwide' : {'compute_qa': True}
    }
    
    p = Parameters(**pv)
    
    os.chdir('/home/inferis/spr/IR/apr21')
    
    f = open('config-d.txt', 'w+')
    try:
        json.dump(p, f, default=to_json, encoding='utf-8', indent=2)
    finally:
        f.close()
    
    main(['-d', '--run', 'direct_imaging', 'config-d.txt'])
