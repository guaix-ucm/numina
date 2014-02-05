#
# Copyright 2008-2014 Universidad Complutense de Madrid
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

'''
Basic Data Products
'''

import warnings

from astropy.io import fits

class DataFrame(object):
    def __init__(self, frame=None, filename=None, itype='UNKNOWN'):
        if frame is None and filename is None:
            raise ValueError('only one in frame and filename can be None') 
        self.frame = frame
        self.filename = filename
        self.itype = itype

    def open(self):
        if self.frame is None:
            return fits.open(self.filename, memmap=True, mode='readonly')
        else:
            return self.frame

    def __getstate__(self):
        if self.frame is None and self.filename is None:
            raise ValueError('only one in frame and filename can be None') 
        # save fits file
        if self.frame is None: 
            # assume filename contains a FITS file
            return {'filename': self.filename}
        else:
            if self.filename:
                filename = self.filename
            elif 'FILENAME' in self.frame[0].header:
                filename = self.frame[0].header['FILENAME']
            else:
                filename = 'result.fits'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.frame.writeto(filename, clobber=True)

        return {'filename': filename, 'itype': self.itype}
    
    @property
    def label(self):
        return self.filename

    def __setstate__(self, state):
        self.filename = state['filename']
        self.itype = state['itype']
        
    def __repr__(self):
        if self.frame is None:
            return "DataFrame(filename=%r)" % self.filename
        elif self.filename is None:
            return "DataFrame(frame=%r)" % self.frame
        else:
            return "DataFrame(filename=%r, frame=%r)" % (self.filename, self.frame)
