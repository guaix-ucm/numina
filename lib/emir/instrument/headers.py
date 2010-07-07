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

try:
    from pkgutil import get_data
except ImportError:
    from numina.compatibility import get_data

from simplejson import loads
from pyfits import Header, Card

from numina.image.storage import FITSCreator
from numina.jsonserializer import deunicode_json

_result_types = ['image', 'spectrum']
_extensions = ['primary', 'variance', 'map', 'wcs']

_all_headers = deunicode_json(loads(get_data('emir.instrument','headers.json')))
        
def _merge(headers, result_type):
    if result_type not in _result_types:
        raise TypeError('Image type not "image" or "spectrum"') 

    result = {}
    for ext in _extensions:
        final = dict(headers['common'][ext])
        final.update(headers[result_type][ext])
        rr = []
        for key, (val, comment) in final.iteritems():
            rr.append(Card(key, val, comment))
        result[ext] = Header(rr)
    
    return result

_image_fits_headers = _merge(_all_headers, 'image')
_spectrum_fits_headers = _merge(_all_headers, 'spectrum')

default = dict(image=_image_fits_headers,
               spectrum=_spectrum_fits_headers)


class EmirImageCreator(FITSCreator):
    '''Builder of Emir direct image.'''
    def __init__(self): 
        super(EmirImageCreator, self).__init__(_image_fits_headers)
        
class EmirSpectrumCreator(FITSCreator):
    '''Builder of Emir spectrum.'''
    def __init__(self): 
        super(EmirSpectrumCreator, self).__init__(_spectrum_fits_headers)
    