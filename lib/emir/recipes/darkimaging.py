#
# Copyright 2008-2009 Sergio Pascual
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

'''Dark image recipe and associates'''

import logging

import pyfits
import scipy.stsci.image as im

from numina import RecipeBase
#import emir.image.combine as com

__version__ = "$Revision$"

_logger = logging.getLogger("emir.recipes")
        
_usage_string = "usage: %prog [options] recipe [recipe-options]"

class DarkImagingResult:
    def __init__(self, image, filename):
        self.hdulist = image
        self.filename = filename
        
    def store(self):
        _logger.debug('Saving %s' % self.filename)
        self.hdulist.writeto(self.filename)

class DarkImaging(RecipeBase):
    '''Recipe to process data taken in Dark current image Mode.
    
    Here starts the long description...
    It continues several lines
    
    '''
    def __init__(self):
        super(DarkImaging, self).__init__(optusage=_usage_string)
        # Default values. This can be read from a file
        self.iniconfig.add_section('inputs')
        self.iniconfig.set('inputs', 'files', '')
        self.iniconfig.add_section('output')
        self.iniconfig.set('output', 'filename', 'output.fits')
                
    def process(self):
        pfiles = self.iniconfig.get('inputs','files')
        pfiles = ' '.join(pfiles.splitlines()).replace(',',' ').split()
        if len(pfiles) == 0:
            _logger.warning('No files to process')
            return
        
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
        
        _logger.debug('We have %d images', len(images))
        # Data from the primary extension
        data = [i['primary'].data for i in images]
        #
        result = im.average(data)
        # Creating the result pyfits structure
        # Creating the primary HDU
        dhdu = pyfits.PrimaryHDU(result)
        # Variance and exposure extensions
        vhdu = pyfits.ImageHDU(name='VARIANCE')
        nhdu = pyfits.ImageHDU(name='NUMBER')
        # Final structure
        hdulist = pyfits.HDUList([dhdu, vhdu, nhdu])
        
        filename = self.iniconfig.get('output','filename')
        return DarkImagingResult(hdulist, filename)
        