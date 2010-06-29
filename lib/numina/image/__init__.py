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

import shutil
import copy

import numpy
import pyfits

class Image(object):
    def __init__(self, filename):
        super(Image, self).__init__()
        self.data = None
        self.meta = None
        self.filename = filename
        self._open = False
        self.hdulist = None

    def is_open(self):
        return self._open

    def open(self, mode='copyonwrite', memmap=False):
        if not self.is_open():
            self._open = True
            self.hdulist = pyfits.open(self.filename, memmap=memmap, mode=mode)
            self.data = self.hdulist['primary'].data
            self.meta = self.hdulist['primary'].header
        
    def close(self, output_verify='exception'):
        if self.is_open():
            self.hdulist['primary'].data = self.data
            self.hdulist.close(output_verify=output_verify)
        self._open = False
        self.data = None
        self.hdulist = None
        
    def __copy__(self):
        new = Image(filename=self.filename)
        return new

    def __str__(self):
        return 'Image(filename="%s")' % (self.filename)
    
    def __getstate__(self):
        return dict(data=None, meta=None, filename=self.filename,
                    _open=False, hdulist=None)
        
    def copy(self, dst):
        shutil.copy(self.filename, dst)
        newimg = copy.copy(self)
        newimg.filename = dst
        return newimg

def compute_median(img, mask, region):
    d = img.data[region]
    m = mask.data[region]
    value = numpy.median(d[m == 0])
    return value, img