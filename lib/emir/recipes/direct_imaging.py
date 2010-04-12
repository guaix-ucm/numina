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

'''Recipe for the reduction of imaging mode observations.

Recipe to reduce observations obtained in imaging mode, considering different
possibilities depending on the size of the offsets between individual images.
In particular, the following strategies are considered: stare imaging, nodded
beamswitched imaging, and dithered imaging. 

A critical piece of information here is a table that clearly specifies which
images can be labeled as *science*, and which ones as *sky*. Note that some
images are used both as *science* and *sky* (when the size of the targets are
small compared to the offsets).

**Observing modes:**
 
 * StareImage
 * Nodded/Beam-switched images
 * Dithered images 


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


import logging

import pyfits
import numpy as np

import numina.recipes as nr
#from numina.exceptions import RecipeError
from numina.image.processing import DarkCorrector, NonLinearityCorrector, FlatFieldCorrector
from numina.image.processing import generic_processing
from numina.array.combine import median
from emir.instrument.headers import EmirImageCreator
import numina.qa as QA

_logger = logging.getLogger("emir.recipes")

class ParameterDescription(nr.ParameterDescription):
    def __init__(self):
        inputs = {'images': [],
                                            'masks': [],
                                            'offsets': [],
                                            'master_bias': '',
                                            'master_dark': '',
                                            'master_flat': '',
                                            'master_bpm': ''}
        optional = {'linearity': [1.0, 0.0],
                  'extinction': 0
                  }
        super(ParameterDescription, self).__init__(inputs, optional)

def combine_shape(shapes, offsets):
    # Computing final image size and new offsets
    sharr = np.asarray(shapes)
    
    offarr = np.asarray(offsets)        
    ucorners = offarr + sharr
    ref = offarr.min(axis=0)        
    finalshape = ucorners.max(axis=0) - ref 
    offsetsp = offarr - ref
    return (finalshape, offsetsp)

class Result(nr.RecipeResult):
    '''Result of the imaging mode recipe.'''
    def __init__(self, qa, result):
        super(Result, self).__init__(qa)
        self.products['result'] = result   
     
class Recipe(nr.RecipeBase):
    '''Recipe to process data taken in imaging mode.
     
    '''
    def __init__(self):
        super(Recipe, self).__init__()
        self.images = []
        self.masks = []
        self.baseshape = ()        

        self.book_keeping = {}
        self.base = 'emir_%s.base'
        self.basemask = 'emir_%s.mask'
        self.iter = 'emir_%s.iter.%02d'
        self.itermask = 'emir_%s.mask.iter.%02d'
        self.iteromask = 'emir_%s.omask.iter.%02d'
        self.inter = 'emir_intermediate.%02d.fits'
        #
        # To be from inputs read later
        self.extinction = 0
        self.airmass_keyword = 'AIRMASS'
        self.airmasses = []
        
    def setup(self, param):
        super(Recipe, self).setup(param)
        self.images = self.inputs['images'].keys()
        self.images.sort()
        self.masks = [self.inputs['images'][k][0] for k in self.images]
        self.extinction = param.optional['extinction']
        
        self.input_checks()

        for i, m in zip(self.images, self.masks):
            self.book_keeping[i] = dict(mask=m, version=i, omask=m, region=None)
            
        
        
    def input_checks(self):
        
        _logger.info("Checking shapes of inputs")
        # Getting the shapes of all the images
        
        image_shapes = []
        
        for fname in self.images:
            hdulist = pyfits.open(fname)
            try:
                _logger.debug("Opening image %s", fname)
                image_shapes.append(hdulist['primary'].data.shape)
                _logger.debug("Shape of image %s is %s", fname, image_shapes[-1])
                self.airmasses.append(hdulist['primary'].header[self.airmass_keyword])
            finally:
                hdulist.close()
        
        mask_shapes = []
        
        for fname in self.masks:
            hdulist = pyfits.open(fname)
            try:
                _logger.debug("Opening mask %s", fname)
                mask_shapes.append(hdulist['primary'].data.shape)
                _logger.debug("Shape of mask %s is %s", fname, mask_shapes[-1])
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

        def newslice(offset, baseshape):
            # this function is almost equivalent to
            # subarray_match
            ndim = len(offset)
            return tuple(slice(offset[i], offset[i] + baseshape[i]) for i in xrange(ndim))

        
        def resize_images(finalshape, offsetsp):
            
            # Resize images
            for fname, offset in zip(self.images, offsetsp):
                hdulist = pyfits.open(fname, mode='readonly')
                try:
                    p = hdulist['primary']
                    newdata = np.zeros(finalshape, dtype=p.data.dtype)
                    region = newslice(offset, self.baseshape)
                    newdata[region] = p.data
                    newfile = self.base % fname
                    pyfits.writeto(newfile, newdata, p.header,
                                   output_verify='silentfix', clobber=True)
                    _logger.info('Resized image %s, offset %s, new shape %s', fname, offset, finalshape)
                    self.book_keeping[fname]['region'] = region
                    self.book_keeping[fname]['version'] = newfile    
                finally:
                    hdulist.close()
    
            # Resize masks
            for fname, base in zip(self.masks, self.images):
                hdulist = pyfits.open(fname, mode='readonly')
                try:
                    p = hdulist['primary']
                    newdata = np.ones(finalshape, dtype=p.data.dtype)
                    region = self.book_keeping[base]['region']
                    newdata[region] = p.data
                    newfile = self.basemask % base
                    pyfits.writeto(newfile, newdata, p.header,
                                   output_verify='silentfix', clobber=True)
                    _logger.info('Resized image %s into image %s, new shape %s',
                                 fname, newfile, finalshape)
                    
                    self.book_keeping[base]['mask'] = newfile    
                finally:
                    hdulist.close()                

        def compute_sky_simple(fname,iternr):            
            # The sky images corresponding to fname
            sky_images = self.inputs['images'][fname][2]
            r_sky_images = [self.book_keeping[i]['version'] for i in sky_images]
            r_sky_masks = [self.book_keeping[i]['omask'] for i in sky_images]
            r_sky_regions = [self.book_keeping[i]['region'] for i in sky_images]
            _logger.info('Iter %d, sky images for %s are %s', itern, fname, sky_images)
            
            if len(sky_images) == 1:
                pass
            
            try:       
                fd = []
                data = []
                masks = []
                
                for fname, region in zip(r_sky_images, r_sky_regions):
                    _logger.debug('Opening %s', fname)
                    hdulist = pyfits.open(fname, memmap=True, mode='readonly')                    
                    data.append(hdulist['primary'].data[region])
                    fd.append(hdulist)
                
                for fname, region in zip(r_sky_masks, r_sky_regions):
                    _logger.debug('Opening %s', fname)
                    hdulist = pyfits.open(fname, memmap=True, mode='readonly')                    
                    masks.append(hdulist['primary'].data[region])
                    fd.append(hdulist)
                
                
                if len(sky_images) == 1:
                    skyval = data[0]
                    skymask = masks == 0
                    median_sky = np.median(skyval[skymask])
                else:
                    # Combine the sky images with masks
                    _logger.debug("Combine the sky images with masks")         
                    skyback = median(data, masks)
                    skyval = skyback[0]
                    skymask = skyback[2] != 0
                    median_sky = np.median(skyval[skymask])
            finally:
                _logger.debug("Closing the sky files")
                map(pyfits.HDUList.close, fd)
            
            _logger.info('Iter %d, median sky background %f', itern, median_sky)
            return median_sky
        
        def compute_superflat(itern):
            # The sky images corresponding to fname
            _logger.info('Iter %d, computing superflat', itern)
            fimages = [self.base % i for i in self.images]
            fmasks = [self.book_keeping[i]['mask'] for i in self.images]
            regions = [self.book_keeping[i]['region'] for i in self.images]
            
            try:       
                fd = []
                data = []
                masks = []
                
                for fname, region in zip(fimages, regions):
                    _logger.debug('Opening %s', fname)
                    hdulist = pyfits.open(fname, memmap=True, mode='readonly')  
                    data.append(hdulist['primary'].data[region])
                    fd.append(hdulist)
                
                for fname, region in zip(fmasks, regions):
                    _logger.debug('Opening %s', fname)
                    hdulist = pyfits.open(fname, memmap=True, mode='readonly')                    
                    masks.append(hdulist['primary'].data[region])
                    fd.append(hdulist)
                _logger.info("Iter %d, computing the median of the images", itern)        
                scales = [np.median(d[m == 0]) for d, m in zip(data, masks)]
                
                # Combine the sky images with masks
                _logger.info("Iter %d, combining the images without offsets", itern)            
                superflat = median(data, masks, scales=scales)
                pyfits.writeto('superflat.fits.itern.%02d' % itern, superflat[0], clobber=True)

            finally:
                _logger.debug("Closing the files")
                map(pyfits.HDUList.close, fd)
            
            return superflat    
                
        def superflat_processing(superflat, itern):
            # TODO: handle masks
            # The sky images corresponding to fname
            
            flat = superflat[0]
            mask = (flat == 0)
            flat[mask] = 1

            regions = [self.book_keeping[i]['region'] for i in self.images]
            
            for fname, region in zip(self.images, regions):
                bfile = self.base % fname
                _logger.debug('Opening %s', bfile)
                hdulist = pyfits.open(bfile, mode='readonly')
                try:
                    _logger.info('Iter %d, processing image %s with superflat', itern, (bfile))
                    newdata = np.zeros_like(hdulist['primary'].data)
                    newdata[region] = hdulist['primary'].data[region] / flat
                    newfile = self.iter % (fname, itern)
                    _logger.debug("Saving fname %s", newfile)
                    self.book_keeping[fname]['version'] = newfile
                    pyfits.writeto(newfile, newdata, header=hdulist['primary'].header,
                                   clobber=True)
                finally:
                    hdulist.close()

        def mask_merging(fname, obj_mask, itern):
            mask = self.book_keeping[fname]['mask']
            hdulist = pyfits.open(mask, mode='readonly')
            try:
                p = hdulist['primary']
                newdata = (p.data != 0) | (obj_mask != 0)
                newdata = newdata.astype('int')
                newfile = self.iteromask % (fname, itern)
                pyfits.writeto(newfile, newdata, p.header,
                               output_verify='silentfix', clobber=True)
                self.book_keeping[fname]['omask'] = newfile
                _logger.info('Iter %d, mask %s merged with object mask into %s', itern, fname, newfile)
            finally:
                hdulist.close() 
        
        def combine_images(backgrounds, itern):
            
            # Combine
            r_images = [self.book_keeping[i]['version'] for i in self.images]
            r_masks = [self.book_keeping[i]['mask'] for i in self.images]
            try:
                fd = []
                data = []
                masks = []
            
                for fname in r_images:
                    _logger.debug('Opening %s', fname)
                    hdulist = pyfits.open(fname, memmap=True, mode='readonly')
                    data.append(hdulist['primary'].data)
                    _logger.debug('Append fits handle %s', hdulist)
                    fd.append(hdulist)
                
                for fname in r_masks:
                    _logger.debug('Opening %s', fname)
                    hdulist = pyfits.open(fname, memmap=True, mode='readonly')
                    masks.append(hdulist['primary'].data)
                    _logger.debug('Append fits handle %s', hdulist)
                    fd.append(hdulist)

                _logger.info('Iter %d, combining images', itern)
                extinc = [pow(10, 0.4 * i * self.extinction)  for i in self.airmasses]
                final_data = median(data, masks, zeros=backgrounds, scales=extinc, dtype='float32')
            finally:
                # One liner
                _logger.debug("Closing the data files")
                map(pyfits.HDUList.close, fd)
                        
            return final_data            
        
        save_intermediate = True
        # Final result constructor
        fc = EmirImageCreator()
        
        # dark correction
        # open the master dark
        dark_data = pyfits.getdata(self.inputs['master_dark'])    
        flat_data = pyfits.getdata(self.inputs['master_flat'])
        
        corrector1 = DarkCorrector(dark_data)
        corrector2 = NonLinearityCorrector(self.optional['linearity'])
        corrector3 = FlatFieldCorrector(flat_data)
        
        generic_processing(self.images, [corrector1, corrector2, corrector3], backup=True)
        
        del dark_data
        del flat_data    

        # Getting the offsets
        offsets = [self.inputs['images'][k][1] for k in self.images]
        #offsets = [(0,0) for k in self.images]
                
        # Computing the shape of the final image
        finalshape, offsetsp = combine_shape(self.baseshape, offsets)
        _logger.info("Shape of the final image %s", finalshape)
        
        # Resize images and masks
        resize_images(finalshape, offsetsp)
        
        for k in self.book_keeping:
            self.book_keeping[k]['omask'] = self.book_keeping[k]['mask']
        
        # Iterate 4 times
        iterations = 1
        
        for itern in xrange(0, iterations):
            _logger.info('Starting iteration %s', itern)
            
            # Super flat
            superflat = compute_superflat(itern)
            
            superflat_processing(superflat, itern)
            
            # TODO Only for images that are Science images
            _logger.info('Iter %d, sky subtraction', itern)     
            sky_backgrounds = map(lambda f : compute_sky_simple(f, itern), self.images)
        
            # Combine
            final_data = combine_images(sky_backgrounds, itern)
        
            if save_intermediate:
                newfile = self.inter % itern
                final = fc.create(final_data[0])
                final.writeto(newfile, output_verify='silentfix', clobber=True)
        
            # Generate object mask (using sextractor)
            _logger.info('Iter %d, generating objects mask', itern)
            obj_mask = sextractor_object_mask(final_data[0], itern)
        
            _logger.info('Iter %d, merging object mask with masks', itern)
            map(lambda f: mask_merging(f, obj_mask, itern), self.images)
            
        for itern in xrange(iterations, iterations + 1):
            _logger.info('Starting iteration %s', itern)
            
            # Compute sky from the image (median)
            # TODO Only for images that are Science images
            _logger.info('Iter %d, sky subtraction', itern)
            sky_backgrounds = map(lambda f : compute_sky_simple(f, itern), self.images)
        
            # Combine
            final_data = combine_images(sky_backgrounds, itern)    
        
        
        _logger.info('Finished iterations')
        
        final = fc.create(final_data[0], extensions=[('variance', final_data[1], None),
                                                     ('map', final_data[2].astype('int16'), None)])
        _logger.info("Final image created")
        
        return Result(QA.UNKNOWN, final)

    
def sextractor_object_mask(array, itern):
    import tempfile
    import subprocess
    import os.path
    
    checkimage = "emir_check%02d.fits" % itern
    
    # A temporary file used to store the array in fits format
    tf = tempfile.NamedTemporaryFile(prefix='emir_', dir='.')
    pyfits.writeto(tf, array)
        
    # Run sextractor, it will create a image called check.fits
    # With the segmentation mask inside
    sub = subprocess.Popen(["sex",
                            "-CHECKIMAGE_TYPE", "SEGMENTATION",
                            "-CHECKIMAGE_NAME", checkimage,
                            '-VERBOSE_TYPE', 'NORMAL',
                             tf.name],
                           stdout=subprocess.PIPE)
    sub.communicate()
    
    segfile = os.path.join('.', 'check.fits')
    
    # Read the segmentation image
    result = pyfits.getdata(segfile)
    
    # Close the tempfile
    tf.close()    
    
    return result
        
    
if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO)
    #_logger.setLevel(logging.INFO)
    import os
    import simplejson as json
    
    from numina.user import main
    from numina.recipes import Parameters
    
    from numina.jsonserializer import to_json
    
    pv = {'inputs' :  { 'images':  
                       {'apr21_0046.fits': ('bpm.fits', (0, 0), ['apr21_0046.fits']),
                        'apr21_0047.fits': ('bpm.fits', (0, 0), ['apr21_0047.fits']),
                        'apr21_0048.fits': ('bpm.fits', (0, 0), ['apr21_0048.fits']),
                        'apr21_0049.fits': ('bpm.fits', (21, -23), ['apr21_0049.fits']),
                        'apr21_0051.fits': ('bpm.fits', (21, -23), ['apr21_0051.fits']),
                        'apr21_0052.fits': ('bpm.fits', (-15, -35), ['apr21_0052.fits']),
                        'apr21_0053.fits': ('bpm.fits', (-15, -35), ['apr21_0053.fits']),
                        'apr21_0054.fits': ('bpm.fits', (-15, -35), ['apr21_0054.fits']),
                        'apr21_0055.fits': ('bpm.fits', (24, 12), ['apr21_0055.fits']),
                        'apr21_0056.fits': ('bpm.fits', (24, 12), ['apr21_0056.fits']),
                        'apr21_0057.fits': ('bpm.fits', (24, 12), ['apr21_0057.fits']),
                        'apr21_0058.fits': ('bpm.fits', (-27, 18), ['apr21_0058.fits']),
                        'apr21_0059.fits': ('bpm.fits', (-27, 18), ['apr21_0059.fits']),
                        'apr21_0060.fits': ('bpm.fits', (-27, 18), ['apr21_0060.fits']),
                        'apr21_0061.fits': ('bpm.fits', (-38, -16), ['apr21_0061.fits']),
                        'apr21_0062.fits': ('bpm.fits', (-38, -16), ['apr21_0062.fits']),
                        'apr21_0063.fits': ('bpm.fits', (-38, -17), ['apr21_0063.fits']),
                        'apr21_0064.fits': ('bpm.fits', (5, 27), ['apr21_0064.fits']),
                        'apr21_0065.fits': ('bpm.fits', (5, 27), ['apr21_0065.fits']),
                        'apr21_0066.fits': ('bpm.fits', (5, 27), ['apr21_0066.fits']),
                        'apr21_0067.fits': ('bpm.fits', (32, -13), ['apr21_0067.fits']),
                        'apr21_0068.fits': ('bpm.fits', (33, -13), ['apr21_0068.fits']),
                        'apr21_0069.fits': ('bpm.fits', (32, -13), ['apr21_0069.fits']),
                        'apr21_0070.fits': ('bpm.fits', (-52, 7), ['apr21_0070.fits']),
                        'apr21_0071.fits': ('bpm.fits', (-52, 8), ['apr21_0071.fits']),
                        'apr21_0072.fits': ('bpm.fits', (-52, 8), ['apr21_0072.fits']),
                        'apr21_0073.fits': ('bpm.fits', (-3, -49), ['apr21_0073.fits']),
                        'apr21_0074.fits': ('bpm.fits', (-3, -49), ['apr21_0074.fits']),
                        'apr21_0075.fits': ('bpm.fits', (-3, -49), ['apr21_0075.fits']),
                        'apr21_0076.fits': ('bpm.fits', (-49, -33), ['apr21_0076.fits']),
                        'apr21_0077.fits': ('bpm.fits', (-49, -32), ['apr21_0077.fits']),
                        'apr21_0078.fits': ('bpm.fits', (-49, -32), ['apr21_0078.fits']),
                        'apr21_0079.fits': ('bpm.fits', (-15, 36), ['apr21_0079.fits']),
                        'apr21_0080.fits': ('bpm.fits', (-16, 36), ['apr21_0080.fits']),
                        'apr21_0081.fits': ('bpm.fits', (-16, 36), ['apr21_0081.fits'])
                        },
                        'master_dark': 'Dark50.fits',
                        'master_flat': 'flat.fits',
                        'master_bpm': 'bpm.fits'
                        },
          'optional' : {'linearity': [1.00, 0.00],
                        'extinction,': 0.05,
                        }          
    }
    
    # Changing the offsets
    # x, y -> -y, -x
    for k in pv['inputs']['images']:
        m, o, s = pv['inputs']['images'][k]
        x, y = o
        o = -y, -x
        pv['inputs']['images'][k] = (m, o, s)
    
    p = Parameters(**pv)
    
    os.chdir('/home/spr/Datos/emir/apr21')
    
    f = open('config-d.json', 'w+')
    try:
        json.dump(p, f, default=to_json, encoding='utf-8', indent=2)
    finally:
        f.close()
            
    main(['--run', 'direct_imaging', 'config-d.json'])
