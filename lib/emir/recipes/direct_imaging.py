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
import uuid
import itertools

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
        self.images = self.parameters.inputs['images'].keys()
        self.images.sort()
        self.masks = [self.parameters.inputs['images'][k][0] for k in self.images]
        self.baseshape = ()
        self.ndim = len(self.baseshape)
        self.input_checks()

        self.book_keeping = {}
        
        for i,m in zip(self.images, self.masks):
            self.book_keeping[i] = dict(masks=[m], versions=[i])
        
        
    def input_checks(self):
        
        _logger.info("Checking shapes of inputs")
        # Getting the shapes of all the images
        
        image_shapes = []
        
        for file in self.images:
            hdulist = pyfits.open(file)
            try:
                _logger.debug("Opening image %s", file)
                image_shapes.append(hdulist['primary'].data.shape)
                _logger.info("Shape of image %s is %s", file, image_shapes.append[-1])
            finally:
                hdulist.close()
        
        mask_shapes = []
        
        for file in self.masks:
            hdulist = pyfits.open(file)
            try:
                _logger.debug("Opening mask %s", file)
                mask_shapes.append(hdulist['primary'].data.shape)
                _logger.info("Shape of mask %s is %s", file, mask_shapes.append[-1])
            finally:
                hdulist.close()

        # All images have the same shape
        self.baseshape = image_shapes[0]
        self.ndim = len(self.baseshape)
        if any(i != self.baseshape for i in image_shapes[1:]):
            _logger.error('Image has shape %s, different to the base shape %s', i, self.baseshape)
            return False
        
        # All masks have the same shape
        if any(i != self.baseshape for i in mask_shapes[1:]):
            _logger.error('Mask has shape %s, different to the base shape %s', i, self.baseshape)
            return False
        
        _logger.info("Shapes of inputs are %s", self.baseshape)
        
        return True
        
    def process(self):

        def resize_image(file, offsetsp, store):
            # Resize images, to final size
            hdulist = pyfits.open(file, memmap=True)
            try:
                p = hdulist['primary']
                newfile = str(uuid.uuid1())
                newdata = np.zeros(finalshape, dtype=p.data.dtype)
                region = tuple(slice(offset[i], offset[i] 
                                     + self.baseshape[i]) for i in xrange(self.ndim))
                newdata[region] = p.data
                pyfits.writeto(newfile, newdata, p.header, output_verify='silentfix', clobber=True)
                _logger.info('Resized image %s into image %s, new shape %s', file, newfile, finalshape)
                self.book_keeping[i][store].append(newfile)
            finally:
                hdulist.close()

        def compute_sky_simple(file):            
            # The sky images corresponding to file
            sky_images = self.parameters.inputs['images'][file][2]
            r_sky_images = [self.book_keeping[i]['versions'][-1] for i in sky_images]
            r_sky_masks = [self.book_keeping[i]['masks'][-1] for i in sky_images]
            _logger.info('Sky images for %s are %s', file, sky_images)
                
            try:       
                fd = []
                data = []
                masks = []
                for file in r_sky_images:
                    hdulist = pyfits.open(file, memmap=True)
                    data.append(hdulist['primary'].data)
                    fd.append(hdulist)
                
                for file in r_sky_masks:
                    hdulist = pyfits.open(file, memmap=True)
                    masks.append(hdulist['primary'].data)
                    fd.append(hdulist)
                
                # Combine the sky images with masks            
                skyback = median(data, masks)
            
                # Close all the memapped files
                # Dont wait for the GC
            finally:
                for f in fd:
                    fd.close()
                
            skyval = skyback[0]
            skymask = skyback[2] != 0
            
            median_sky = np.median(skyval[skymask])
            _logger.info('Median sky value is %d', median_sky)
            return median_sky

        def mask_merging(file, obj_mask):
            
            hdulist = pyfits.open(self.book_keeping[file]['masks'][-1])
            try:
                p = hdulist['primary']
                newdata = (p.data != 0) | (obj_mask != 0)
                newdata = newdata.astype('int')
                newfile = str(uuid.uuid1())
                pyfits.writeto(newfile, newdata, p.header, 
                               output_verify='silentfix', clobber=True)
                self.book_keeping[file]['masks'].append(newfile)
                _logger.info('Mask %s merged with object mask into %s', file, newfile)
            finally:
                hdulist.close() 
        
        def combine_images(backgrounds):
            
            # Combine
            r_images = [self.book_keeping[i]['versions'][-1] for i in self.images]
            r_masks = [self.book_keeping[i]['masks'][-1] for i in self.masks]
            
            try:
                fd = []
                data = []
                masks = []
            
                for file in r_sky_images:
                    hdulist = pyfits.open(file, memmap=True)
                    data.append(hdulist['primary'].data)
                    fd.append(hdulist)
                
                for file in r_sky_masks:
                    hdulist = pyfits.open(file, memmap=True)
                    masks.append(hdulist['primary'].data)
                    fd.append(hdulist)
            
                    final_data = median(r_images, r_masks, zeros=backgrounds)
            finally:
                for f in fd:
                    fd.close()
                # One liner
                map(pyfits.HDUList.close, fd)
            
            _logger.info('Combined images')
            return final_data            
        
        # dark correction
        # open the master dark
        dark_data = pyfits.getdata(self.parameters.inputs['master_dark'])    
        flat_data = pyfits.getdata(self.parameters.inputs['master_flat'])
        
        corrector1 = DarkCorrector(dark_data)
        corrector2 = NonLinearityCorrector(self.parameters.inputs['linearity'])
        corrector3 = FlatFieldCorrector(flat_data)
        
        generic_processing(self.images, [corrector1, corrector2, corrector3], backup=True)
        
        del dark_data
        del flat_data    
        
        # Getting the offsets
        offsets = [self.parameters.inputs['images'][k][1] for k in self.images]
                
        # Computing the shape of the final image
        finalshape, offsetsp = combine_shape(self.baseshape, offsets)
        _logger.info("Shape of the final image %s", finalshape)
        
        # Iteration 0        
        map(lambda f, o: resize(f, o, store='versions'), self.images, offsetsp)
        map(lambda f, o: resize(f, o, store='masks'), self.masks, offsetsp)
                
        # Compute sky from the image (median)
        # TODO: Only for images that are Science images        
        sky_backgrounds = map(compute_sky_simple, self.images)
        
        # Combine
        final_data = combine_images(backgrounds)
        
        # Generate object mask (using sextractor)
        _logger.info('Generated objects mask')
        obj_mask = sextractor_object_mask(final_data[0])
        
        _logger.info('Object mask merged with masks')
        map(mask_merging, self.files)
        
        # Iterate 4 times
        iterations = 4
        
        for iter in xrange(1, iterations + 1):
            _logger.info('Starting iteration %s', iter)
            
            # Compute sky from the image (median)
            # TODO Only for images that are Science images        
            sky_backgrounds = compute_sky_simple(self.images)
        
            # Combine
            final_data = combine_images(backgrounds)
        
            # Generate object mask (using sextractor)
            _logger.info('Generated objects mask')
            obj_mask = sextractor_object_mask(final_data[0])
        
            _logger.info('Object mask merged with masks')
            mask_merging(self.images, obj_mask)
        
        _logger.info('Finished iterations')
        
        fc = EmirImage()
        
        final = fc.create(final_data[0])
        _logger.info("Final image created")
        
        return Result(QA.UNKNOWN, final)

    
def sextractor_object_mask(array):
        import tempfile
        import subprocess
        import shutil
        
        # Creating a temporary directory
        tmpdir = tempfile.mkdtemp()
        
        # A temporary file used to store the array in fits format
        tf = tempfile.NamedTemporaryFile(dir=tmpdir)
        pyfits.writeto(tf, array)
        
        # Copying a default.param file
        sub = subprocess.Popen(["cp", "/home/spr/devel/workspace/sextractor/config/default.param", 
                                tmpdir], stdout=subprocess.PIPE)
        sub.communicate()
        
        # Copying a default.conv
        sub = subprocess.Popen(["cp", "/usr/share/sextractor/default.conv",  tmpdir], 
                               stdout=subprocess.PIPE)
        sub.communicate()
        
        # Run sextractor, it will create a image called check.fits
        # With the segmentation mask inside
        sub = subprocess.Popen(["sex", "-CHECKIMAGE_TYPE", "SEGMENTATION", tf.name],
                               stdout=subprocess.PIPE, cwd=tmpdir)
        result = sub.communicate()
        
        segfile = os.path.join(tmpdir, 'check.fits')
        
        # Read the segmentation image
        result = pyfits.getdata(segfile)
        
        # Close the tempfile
        tf.close()
        # Remove everything
        # Inside the temporary directory
        shutil.rmtree(tmpdir)
        
        return result
        
    
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
    
    os.chdir('/home/spr/Datos/IR/apr21')
    
    f = open('config-d.txt', 'w+')
    try:
        json.dump(p, f, default=to_json, encoding='utf-8', indent=2)
    finally:
        f.close()
    
    main(['-d', '--run', 'direct_imaging', 'config-d.txt'])
    
    
    
    
    
    
    
