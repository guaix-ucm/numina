#
# Copyright 2008-2012 Universidad Complutense de Madrid
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

import pyfits

class DataFrame(object):
    def __init__(self, frame=None, filename=None):
        if frame is None and filename is None:
            raise ValueError('only one in frame and filename can be None') 
        self.frame = frame
        self.filename = filename

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
            elif self.frame[0].header.has_key('FILENAME'):
                filename = self.frame[0].header['FILENAME']
            else:
                filename = 'result.fits'
            self.frame.writeto(filename, clobber=True)

        return {'filename': filename}

    def __setstate__(self, state):
        # FIXME: this is not exactly what we had in the begining...
        self.frame = pyfits.open(state['filename'])
        self.filename = state['filename']

    def __repr__(self):
        if self.frame is None:
            return "DataFrame(filename=%r)" % self.filename
        elif self.filename is None:
            return "DataFrame(frame=%r)" % self.frame
        else:
            return "DataFrame(filename=%r, frame=%r)" % (self.filename, self.frame)
