#
# Copyright 2008-2011 Sergio Pascual
# 
# This file is part of Numina
# 
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
# 

import shutil
import copy

import numpy # pylint: disable-msgs=E1101
import pyfits

from ..array import resize_array 

class DiskImage(object):
    def __init__(self, filename):
        super(DiskImage, self).__init__()
        self.data = None
        self.meta = None
        self.filename = filename
        self._open = False
        self._open_extension = 'primary'
        self.hdulist = None

    def is_open(self):
        return self._open

    def open(self, mode='copyonwrite', memmap=False, extension='primary'):
        if not self.is_open():
            self.hdulist = pyfits.open(self.filename, memmap=memmap, mode=mode)
            self.data = self.hdulist[extension].data
            self.meta = self.hdulist[extension].header
            self._open = True
            self._open_extension = extension
        
    def close(self, output_verify='exception'):
        if self.is_open():
            self.hdulist[self._open_extension].data = self.data
            self.hdulist.close(output_verify=output_verify)
        self._open = False
        self._open_extension = 'primary'
        self.data = None
        self.hdulist = None
        
    def __copy__(self):
        new = DiskImage(filename=self.filename)
        return new

    def __str__(self):
        return 'DiskImage(filename="%s")' % (self.filename)
    
    def __getstate__(self):
        return dict(filename=self.filename)
                
    def __setstate__(self, data):
        self.__init__(**data)
        
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

def get_image_shape(header):
    ndim = header['naxis']
    return tuple(header.get('NAXIS%d' % i) for i in range(1, ndim + 1))

def resize_hdu(hdu, newshape, region, fill=0.0):
    basedata = hdu.data
    newdata = resize_array(basedata, newshape, region, fill=fill)                
    newhdu = pyfits.PrimaryHDU(newdata, hdu.header)                
    return newhdu

def resize_fits(fitsfile, newfilename, newshape, region, fill=0.0, clobber=True):
    
    close_on_exit = False
    if isinstance(fitsfile, basestring):
        hdulist = pyfits.open(fitsfile, mode='readonly')
        close_on_exit = True
    else:
        hdulist = fitsfile
        
    try:
        hdu = hdulist['primary']
        newhdu = resize_hdu(hdu, newshape, region, fill=fill)
        newhdu.writeto(newfilename, clobber=clobber)
    finally:
        if close_on_exit:
            hdulist.close()



