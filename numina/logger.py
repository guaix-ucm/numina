#
# Copyright 2008-2011 Universidad Complutense de Madrid
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


'''Extra logging handlers for the numina logging system.'''

import logging

import pyfits

from .recipes import Image

class FITSHistoryHandler(logging.Handler):
    '''Logging handler using HISTORY FITS cards'''
    def __init__(self, header):
        logging.Handler.__init__(self)
        self.header = header

    def emit(self, record):
        msg = self.format(record)
        self.header.add_history(msg)

def log_to_history(logger):
    '''Decorate function, adding a logger handler stored in FITS.'''

    def log_to_history_decorator(method):

        def l2h_method(self, block):
            history_header = pyfits.Header()

            fh =  FITSHistoryHandler(history_header)
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)

            try:
                result = method(self, block)
                if 'products' in result:
                    for r in result['products']:
                       if isinstance(r, Image):
                           hdr = r.image[0].header
                           hdr.ascardlist().extend(history_header.ascardlist())
                return result 
            finally:
                logger.removeHandler(fh)
        return l2h_method

    return log_to_history_decorator
