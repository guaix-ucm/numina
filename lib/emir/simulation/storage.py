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
import os
import pickle
import logging

import pyfits

__version__ = "$Revision$"

# Classes are new style
__metaclass__ = type

_logger = logging.getLogger("emir.storage")

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
        