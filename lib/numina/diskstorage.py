#
# Copyright 2008-2009 Sergio Pascual
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

# $Id$

__version__ = "$Revision$"

import simplejson as json

from pyfits.NP_pyfits import HDUList

def store_to_disk(result, filename='products.json'):
    rep = {}
    for key, val in result.products.iteritems():
        if isinstance(val, HDUList):
            #where = val[0].header.get('filename')
            where = None
            if not where:
                where = '%s.fits' % key
            val.writeto(where, clobber=True, output_verify='ignore')
        else:
            rep[key] = val
    
    f = open(filename, 'w+') 
    try:
        json.dump(rep, f)
    finally:
        f.close()
