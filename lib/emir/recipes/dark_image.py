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

'''Dark image recipe.

Recipe to process dark images. The dark images will be combined 
weighting with the inverses of the corresponding variance extension. 
They do not have to be of the same exposure time t, they will be 
scaled to the same t0 ~ 60s (those with very short exposure time 
should be avoided). 

**Inputs:**

 * A list of dark images 
 * A model of the detector (gain, RN)

**Outputs:**

 * A combined dark frame, with variance extension and quality flag. 

**Procedure:**

The process will remove cosmic rays (using a typical sigma-clipping algorithm).

''' 

__version__ = "$Revision$"
# $Source$

import logging

import pyfits
import scipy.stsci.image as im

from numina.recipes import RecipeBase, RecipeResult
from numina.exceptions import RecipeError
from numina.image.storage import FITSCreator
from emir.simulation.headers import default_fits_headers


_logger = logging.getLogger("emir.recipes")
        
_usage_string = "usage: %prog [options] recipe [recipe-options]"

class Result(RecipeResult):
    '''Result of the DarkImaging recipe.'''
    def __init__(self, image, filename):
        super(Result, self).__init__()
        self.hdulist = image
        self.filename = filename
        
    def store(self):
        _logger.debug('Saving %s' % self.filename)
        self.hdulist.writeto(self.filename)

class Recipe(RecipeBase):
    '''Recipe to process data taken in Dark current image Mode.
    
    Here starts the long description...
    It continues several lines
    
    '''
    def __init__(self):
        super(Recipe, self).__init__(optusage=_usage_string)
        # Default values. This can be read from a file
        self.iniconfig.add_section('inputs')
        self.iniconfig.set('inputs', 'files', '')
        self.iniconfig.add_section('output')
        self.iniconfig.set('output', 'filename', 'output.fits')
        self.creator = FITSCreator(default_fits_headers)
        
    def process(self):
        pfiles = self.iniconfig.get('inputs', 'files')
        pfiles = ' '.join(pfiles.splitlines()).replace(',', ' ').split()
        if len(pfiles) == 0:
            raise RecipeError("No files to process")
        
        images = []
        try:
            for i in pfiles:
                _logger.debug('Loading %s', i)
                images.append(pyfits.open(i))
        except IOError, err:
            _logger.error(err)
            _logger.debug('Cleaning up hdus')
            for i in images:
                i.close()            
            raise RecipeError(err)
        
        _logger.debug('We have %d images', len(images))
        # Data from the primary extension
        data = [i['primary'].data for i in images]
        #
        result = im.average(data)
  
        # Final structure
        extensions = [('VARIANCE', None, None), ('NUMBER', None, None)]
        hdulist = self.creator.create(result, None, extensions)
        
        filename = self.iniconfig.get('output', 'filename')
        
        return Result(hdulist, filename)
        
