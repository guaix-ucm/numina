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

import simplejson as json

from pyfits.NP_pyfits import HDUList

from generic import generic

@generic
def store(obj, where=None):
    raise TypeError(repr(obj) + ' is not storable')

@store.register(HDUList)
def _store_fits(obj, where='file.fits'):
    obj.writeto(where, clobber=True, output_verify='ignore')
 
@store.register(type({}))
def _store_map(obj, where='products.json'):
    f = open(where, 'w+') 
    try:
        json.dump(obj, f)
    finally:
        f.close()

#def store_to_disk(obj):
#    if hasattr(obj, 'products'):
#        for key, val in obj.__dict__.iteritems():
#            store(val, key)
#    else:
#        raise TypeError(repr(obj) + ' is not storable')
