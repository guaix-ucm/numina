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

# $Id$

__version__ = "$Revision$"

import os
from os.path import join as pjoin
from cPickle import dump, load
import logging

# Classes are new style
__metaclass__ = type

_logger = logging.getLogger("numina.storage")

class RunCounter:
    '''Persistent run number counter'''
    def __init__(self, template, ext='.fits',
                 directory=".", pstore='index.pkl', last=1):
        self.template = template
        self.ext = ext
        self.directory = directory
        self.pstore = pstore
        self.last = last
        self.complete = pjoin(self.directory, self.template + self.ext)
        _logger.debug('Accessing image directory: %s' % self.directory)
        if not os.access(self.directory, os.F_OK):
            _logger.debug('Creating image directory %s' % self.directory)
            os.mkdir(self.directory)
        try:
            pkl_file = open(self.pstore, 'rb')
            try:
                _logger.debug('Loading status in %s' % self.pstore)
                self.last = load(pkl_file)
            finally:
                pkl_file.close() 
        except IOError, strrerror:            
            _logger.error(strrerror)
                
    def store(self):
        try:
            pkl_file = open(self.pstore, 'wb')
            try:                
                dump(self.last, pkl_file)
                _logger.debug('Storing internal status in %s' % self.pstore)
            finally:
                pkl_file.close()
        except IOError, strrerror:            
            _logger.error(strrerror)
            
    def runstring(self):
        '''Return the run number and the file name.'''
        run = self.template % self.last
        cfile = self.complete % self.last
        self.last += 1
        return (run, cfile)
        
