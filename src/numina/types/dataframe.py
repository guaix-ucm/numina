#
# Copyright 2008-2014 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
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
            return fits.open(self.filename, mode='readonly')
        else:
            return self.frame

    @property
    def label(self):
        return self.filename

    def __repr__(self):
        if self.frame is None:
            return f"DataFrame(filename={self.filename!r})"
        elif self.filename is None:
            return f"DataFrame(frame={self.frame!r})"
        else:
            fmt = "DataFrame(filename=%r, frame=%r)"
            return fmt % (self.filename, self.frame)

    def __numina_load__(self, obj):
        if obj is None:
            return None
        else:
            return DataFrame(filename=obj)