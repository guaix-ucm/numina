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

from __future__ import with_statement
import datetime
import pyfits
import os
import pickle
import logging

logger = logging.getLogger("emir.fits_storage")

class FitsStorage:
    longname = 'fits storage'
    def __init__(self, default_fits_headers, filename, directory, index):
        self.filename = filename
        self.default_fits_headers = default_fits_headers
        self.dir = directory
        self.pstore = index
        self.last = 0
        self.complete = os.path.realpath(os.path.join(self.dir,self.filename))
        logger.debug('Accessing image dir: %s' % self.dir)
        if not os.access(self.dir, os.F_OK):
            logger.debug('Creating image dir %s' % self.dir)
            os.mkdir(self.dir)
        try:
            with open(self.pstore,'rb') as pkl_file:
                logger.debug('Loading status in %s' % self.pstore)
                self.last = pickle.load(pkl_file)
        except IOError, strrerror:            
            logger.error(strrerror)
                
    def __del__(self):
    	try:
    		with open(self.pstore, 'wb') as pkl_file:                
    		    pickle.dump(self.last, pkl_file)
                logger.debug('Storing status in %s' % self.pstore)
        except IOError, strrerror:            
            logger.error(strrerror)

    def add_fits_primary(self, template_headers, data = None, updateheaders = None):
        hdu = pyfits.PrimaryHDU(data, template_headers)
        header = hdu.header        
        if updateheaders is not None:
            logger.info('Updating keywords in %s header', 'PRIMARY')      
            for key in updateheaders:
                try:
                    logger.debug('Updating keyword %s with value %s', key, updateheaders[key])
                    header[key] = updateheaders[key]
                except KeyError:
                    logger.warning("Keyword %s not permitted IN FITS header", key)
        return hdu
    
    
    
    def store(self, data=None, variance=None, wcs=None, pipeline=None, engineering=None, headers = None):
        if headers is None:
            headers = {'RUN':self.last}
        else:
            headers.update({'RUN':self.last})
        # Primary    
        created_hdus = []
        created_hdus.append(self.add_fits_primary(self.default_fits_headers['primary'], data, updateheaders = headers))
        
        # Updating time in PRIMARY header
        now = datetime.datetime.now()
        nowstr = now.strftime('%FT%T')
        created_hdus[0].header["DATE"] = nowstr
        created_hdus[0].header["DATE-OBS"] = nowstr
        if variance is not None:
            hdu = add_fits_extension('ERROR DATA', variance_headers, data=variance, updateheaders={'DATE':nowstr, 'DATE-OBS':nowstr})
            created_hdus.append(hdu)
                   
        if wcs is not None:
            hdu = add_fits_extension('WCSDVARR', wcs_headers, empty=True)
            created_hdus.append(hdu)
    
        if pipeline is not None:
            hdu = add_fits_extension('PIPELINE', wcs_headers, empty=True)
            created_hdus.append(hdu)

        if engineering is not None:
            hdu = add_fits_extension('ENGINEERING', wcs_headers, empty=True)
            created_hdus.append(hdu)
    
            
        hdulist = pyfits.HDUList(created_hdus)
        logger.info('Writing to disk')
        try:
            hdulist.writeto(self.complete % self.last)
            logger.info('Done %s', (self.filename % self.last))
            self.last += 1
        except IOError, strrerror:
            self.lg.error(strrerror)
