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


from numina.exceptions import RecipeError, ParameterError

class DataProduct(object):
    '''Base class for Recipe Products'''
    pass

class DataFrame(DataProduct):
    def __init__(self, frame):
        self.frame = frame
        self.filename = None

    def __getstate__(self):
        # save fits file
        filename = 'result.fits'
        if self.frame[0].header.has_key('FILENAME'):
            filename = self.frame[0].header['FILENAME']
            self.frame.writeto(filename, clobber=True)

        return {'frame': filename}

    def __setstate__(self, state):
        # this is not exactly what we had in the begining...
        self.frame = pyfits.open(state['frame'])
        self.filename = state['frame']
