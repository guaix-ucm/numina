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

from astropy.io import fits


class DataFrame(object):
    '''A handle to a image in disk or in memory.'''
    def __init__(self, frame=None, filename=None):
        if frame is None and filename is None:
            raise ValueError('only one in frame and filename can be None')
        self.frame = frame
        self.filename = filename

    def open(self):
        if self.frame is None:
            return fits.open(self.filename, memmap=True, mode='readonly')
        else:
            return self.frame

    @property
    def label(self):
        return self.filename

    def __repr__(self):
        if self.frame is None:
            return "DataFrame(filename=%r)" % self.filename
        elif self.filename is None:
            return "DataFrame(frame=%r)" % self.frame
        else:
            fmt = "DataFrame(filename=%r, frame=%r)"
            return fmt % (self.filename, self.frame)
