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
from os.path import join as pjoin
import cPickle
from cPickle import dump
import logging


__version__ = "$Revision$"

# Classes are new style
__metaclass__ = type

_logger = logging.getLogger("emir.storage")

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
            with open(self.pstore, 'rb') as pkl_file:
                _logger.debug('Loading status in %s' % self.pstore)
                self.last = cPickle.load(pkl_file)
        except IOError, strrerror:            
            _logger.error(strrerror)
                
    def store(self):
        try:
            with open(self.pstore, 'wb') as pkl_file:                
                cPickle.dump(self.last, pkl_file)
                _logger.debug('Storing internal status in %s' % self.pstore)
        except IOError, strrerror:            
            _logger.error(strrerror)
            
    def runstring(self):
        '''Return the run number and the file name.'''
        run = self.template % self.last
        cfile = self.complete % self.last
        self.last += 1
        return (run, cfile)
        
