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

_internal_map = {}

def register(cls):
    def wrap(f):
        _internal_map[cls] = f
        def w_f(*args):
            return f(*args)
        
        return w_f
    
    return wrap

def store(obj, where=None):
    if obj.__class__ in _internal_map:
        tstore = _internal_map[obj.__class__]
        tstore(obj, where)
    else:
        raise TypeError(repr(obj) + ' is not storable')

@register(HDUList)
def _store_fits(obj, where='file.fits'):
    obj.writeto(where, clobber=True, output_verify='ignore')
 
@register(type({}))
def _store_map(obj, where='products.json'):
    f = open(where, 'w+') 
    try:
        json.dump(obj, f)
    finally:
        f.close()

_internal_map[type([])] = _store_map

def store_to_disk(obj):
    if hasattr(obj, 'products'):
        for key, val in obj.__dict__.iteritems():
            store(val, key)
    else:
        raise TypeError(repr(obj) + ' is not storable')
