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


import datetime
import logging

import pyfits

__version__ = "$Revision: 410 $"

# Classes are new style
__metaclass__ = type

_logger = logging.getLogger("numina.image")

class FITSCreator:
    '''Builder of multi-extension FITS images.'''
    def __init__(self, default_headers):
        self.defaults = default_headers
        
    def init_primary_HDU(self, data=None, headers=None):
        '''Create the primary HDU of the FITS file.'''
        hdu = pyfits.PrimaryHDU(data, self.defaults['PRIMARY'])
                
        if headers is not None:
            _logger.info('Updating keywords in %s header', 'PRIMARY')      
            for key in headers:
                try:
                    _logger.debug('Updating keyword %s with value %s', 
                                  key, headers[key])
                    hdu.header[key] = headers[key]
                except KeyError:
                    _logger.warning("Keyword %s not permitted in FITS header", key)
        return hdu
    
    def init_extension_HDU(self, data=None, headers=None, extname=None):
        '''Create a HDU extension of the FITS file.'''
        try:
            hdu = pyfits.ImageHDU(data, self.defaults[extname], name=extname)
        except KeyError:
            hdu = pyfits.ImageHDU(data, None, name=extname)
        
        if headers is not None:
            _logger.info('Updating keywords in %s header', extname)
            for key in headers:
                try:
                    _logger.debug('Updating keyword %s with value %s', 
                                  key, headers[key])
                    hdu.header[key] = headers[key]
                except KeyError:
                    _logger.warning("Keyword %s not permitted ins FITS header",
                                    key)    
    
        return hdu

    def create(self, data=None, headers=None, extensions=None):
        
        created_hdus = []
        hdu = self.init_primary_HDU(data, headers=headers)
        created_hdus.append(hdu)
        
        # Updating time
        now = datetime.datetime.now()
        nowstr = now.strftime('%FT%T')
        created_hdus[0].header["DATE"] = nowstr
        created_hdus[0].header["DATE-OBS"] = nowstr
        dateheaders = {'DATE':nowstr, 'DATE-OBS':nowstr}

        if extensions is not None:
            for (ename, edata, eheaders) in extensions:
                if eheaders is not None:
                    eheaders.update(dateheaders)
                hdu = self.init_extension_HDU(edata, eheaders, ename)
                created_hdus.append(hdu)
                        
        hdulist = pyfits.HDUList(created_hdus)
        return hdulist
