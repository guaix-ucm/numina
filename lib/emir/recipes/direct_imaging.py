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

import os.path
import logging

import pyfits
import numpy

from numina.recipes import RecipeBase
from numina.recipes import RecipeResult
#from numina.exceptions import RecipeError

__version__ = "$Revision$"

_logger = logging.getLogger("emir.recipes")

class Result(RecipeResult):
    '''Result of the imaging mode recipe.'''
    def __init__(self):
        super(Result, self).__init__()
        
    def store(self):
        '''Description of store.
        
        :rtype: None'''
        pass


class Recipe(RecipeBase):
    '''Recipe to process data taken in imaging mode.
     
    '''
    def __init__(self, options):
        super(Recipe, self).__init__()
        self.options = options
        
    def process(self):

        options = self.options
        # dark correction
        # open the master dark
        dark_data = pyfits.getdata(options.master_dark)
        # get the data
        for f in options.files:
            # destiny
            newf = self.get_processed(f, 'D')
            if os.path.lexists(newf) and not options.clobber:
                _logger.info('File %s exists, skipping', newf)
                continue
            (file_data, file_header) = pyfits.getdata(f, header=True)
                        
            # subtract the master bias data
            file_data -= dark_data
            
            # save as filename - b .fits
            
            pyfits.writeto(newf, file_data, file_header, 
                           output_verify=options.output_verify, 
                           clobber=options.clobber)
            _logger.info('Processing %s', newf)
            
        
        del dark_data
        
        # non-linearity correction
        for f in options.files:
            # open file            
            # get the data
            newf = self.get_processed(f, 'DL')
            if os.path.lexists(newf) and not options.clobber:
                _logger.info('File %s exists, skipping', newf)
                continue
            
            pf = self.get_processed(f, 'D')
            (file_data, file_header) = pyfits.getdata(f, header=True)
            
            file_data = numpy.polyval(options.linearity, file_data)
            
            pyfits.writeto(newf, file_data, file_header,
                           output_verify=options.output_verify, 
                           clobber=options.clobber)
            _logger.info('Processing %s', newf)
            
        
        flat_data = pyfits.getdata(options.master_flat)
        # non-linearity correction
        for f in options.files:
            # open file            
            # get the data
            newf = self.get_processed(f, 'DLF')
            if os.path.lexists(newf) and not options.clobber:                
                _logger.info('File %s exists, skipping', newf)
                continue
            
            pf = self.get_processed(f, 'DL')
            (file_data, file_header) = pyfits.getdata(pf, header=True)
            
            file_data /= flat_data
            
            pyfits.writeto(newf, file_data, file_header, 
                           output_verify=options.output_verify, 
                           clobber=options.clobber)
            _logger.info('Processing %s', newf)
            
        
        del flat_data    
        
        # Data pre processed
        number_of_iterations = 4
        
        # ** 2 iter for bright objects + sextractor tunning
        # ** 4 iter for dim objects + sextractor tunning
        
        # ** QA after 1st iter
        # * Flux control
        # * Offset refinement

        # first iteration, without segmentation mask
        
        # Compute the initial sky subtracted images
       
        current_iteration = 0
       
        for f in options.files:
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
            
            
            pyfits.writeto(newf, file_data, file_header,
                           output_verify=options.output_verify, 
                           clobber=options.clobber)
            _logger.info('Processing %s', newf)
            
        
        
        return Result()
    
    def get_processed(self, filename, flag):
        ex1 = os.path.splitext(filename)
        oldes = os.path.extsep
        os.path.extsep = '_'
        ex2 = os.path.splitext(ex1[0])
        os.path.extsep = oldes
        prop = ex2[1] 
        if not prop:
            prop = '_'
        prop += flag
        return '%s%s%s' % (ex2[0], prop, ex1[1])
    
    
    def mark_processed(self, filename, flag):
        ex1 = os.path.splitext(filename)
        oldes = os.path.extsep
        os.path.extsep = '_'
        ex2 = os.path.splitext(ex1[0])
        os.path.extsep = oldes
        prop = ex2[1] 
        if not prop:
            prop = '_'
        prop += flag
        return '%s%s%s' % (ex2[0], prop, ex1[1])
    
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
    
            
    class Options:
        pass
    
    options = Options()
    options.files = ['apr21_0046.fits', 'apr21_0047.fits', 'apr21_0048.fits', 
                     'apr21_0049.fits', 'apr21_0050.fits']
    
    options.master_bias = 'mbias.fits'
    options.master_dark = 'Dark50.fits'
    # Higher order first!
    options.linearity = (1e-3, 1e-2, 0.99, 0.00)
    options.master_flat = 'DummyFlat.fits'
    options.clobber = True
    options.master_bpm = 'bpm.fits'
    options.output_verify = 'ignore'
    
    r = Recipe(options)
    r.process() 
