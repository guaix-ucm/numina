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

'''User command line interface of Numina.'''

from numina.generic import generic

from numina.core import DataFrame
from numina.core.types import ListOf
import warnings

@generic
def store(obj, where):
    return obj

@store.register(DataFrame)
def store_df(obj, where):
    # save fits file
    if obj.frame is None:
        # assume filename contains a FITS file
        return None
    else:
        if obj.filename:
            filename = obj.filename
        elif 'FILENAME' in obj.frame[0].header:
            filename = obj.frame[0].header['FILENAME']
        else:
            filename = where.get_next_fits_filename()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            obj.frame.writeto(filename, clobber=True)
        return filename

@store.register(list)
def store_list(obj, where):
    return [store(o, where) for o in obj]

