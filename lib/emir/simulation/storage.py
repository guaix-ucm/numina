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
import os
import pickle
import logging

import pyfits

__version__ = "$Revision$"

# Classes are new style
__metaclass__ = type

_logger = logging.getLogger("emir.storage")

class FITSCreator:
    def __init__(self, default_fits_headers):
        self.defaults = default_fits_headers
        
    def init_primary_HDU(self, data=None, updateheaders=None):
        hdu = pyfits.PrimaryHDU(data, self.defaults['primary'])
        header = hdu.header        
        if updateheaders is not None:
            _logger.info('Updating keywords in %s header', 'PRIMARY')      
            for key in updateheaders:
                try:
                    _logger.debug('Updating keyword %s with value %s', key, updateheaders[key])
                    header[key] = updateheaders[key]
                except KeyError:
                    _logger.warning("Keyword %s not permitted in FITS header", key)
        return hdu
    
    def init_extension_HDU(self, extname, empty=False, data=None, updateheaders=None):
        if data is None or empty:
            hdu = pyfits.ImageHDU()
        else:
            hdu = pyfits.ImageHDU(data)
        
        header = hdu.header
        _logger.info('Creating %s header', extname)
        for key,val,comment in self.defaults[extname.tolower()]:
            header.update(key, val, comment)
        header["EXTNAME"] = extname
        
        _logger.info('Updating keywords in %s header', extname)
        
        if updateheaders is not None:
            for key in updateheaders:
                try:
                    _logger.debug('Updating keyword %s with value %s', key, updateheaders[key])
                    header[key] = updateheaders[key]
                except KeyError:
                    _logger.warning("Keyword %s not permitted IN FITS header", key)    
    
        return hdu
    def create(self, data=None, variance=None, wcs=None, 
              pipeline=None, engineering=None, headers=None):
        
        created_hdus = []
        hdu = self.init_primary_HDU(data, updateheaders=headers)
        created_hdus.append(hdu)
        
        # Updating time in PRIMARY header
        now = datetime.datetime.now()
        nowstr = now.strftime('%FT%T')
        created_hdus[0].header["DATE"] = nowstr
        created_hdus[0].header["DATE-OBS"] = nowstr
        if variance is not None:
            updateheaders={'DATE':nowstr, 'DATE-OBS':nowstr}
            hdu = self.init_extension_HDU('ERROR DATA', self.defaults['variance'], 
                                     data=variance, 
                                     updateheaders=updateheaders)
            created_hdus.append(hdu)
                   
        if wcs is not None:
            hdu = self.init_extension_HDU('WCSDVARR', self.defaults['wcs'], empty=True)
            created_hdus.append(hdu)
    
        if pipeline is not None:
            hdu = self.init_extension_HDU('PIPELINE', self.defaults['pipeline'], empty=True)
            created_hdus.append(hdu)

        if engineering is not None:
            hdu = self.init_extension_HDU('ENGINEERING', self.defaults['engineering'], empty=True)
            created_hdus.append(hdu)
                
        hdulist = pyfits.HDUList(created_hdus)
        return hdulist

class FITSStorage:
    def __init__(self, filename, directory, index):
        self.filename = filename
        self.dir = directory
        self.pstore = index
        self.last = 0
        self.complete = os.path.realpath(os.path.join(self.dir,self.filename))
        _logger.debug('Accessing image dir: %s' % self.dir)
        if not os.access(self.dir, os.F_OK):
            _logger.debug('Creating image dir %s' % self.dir)
            os.mkdir(self.dir)
        try:
            with open(self.pstore,'rb') as pkl_file:
                _logger.debug('Loading status in %s' % self.pstore)
                self.last = pickle.load(pkl_file)
        except IOError, strrerror:            
            _logger.error(strrerror)
                
    def __del__(self):
        try:
            with open(self.pstore, 'wb') as pkl_file:                
                pickle.dump(self.last, pkl_file)
                _logger.debug('Clean up, storing internal status in %s and exiting' % self.pstore)
        except IOError, strrerror:            
            _logger.error(strrerror)
            
    def store(self, hdulist):
        _logger.info('Writing to disk')
        try:
            hdulist.writeto(self.complete % self.last)
            _logger.info('Done %s', (self.filename % self.last))
            self.last += 1
        except IOError, strrerror:
            _logger.error(strrerror)
        