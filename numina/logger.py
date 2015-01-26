#
# Copyright 2008-2015 Universidad Complutense de Madrid
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

from astropy.io import fits


class FITSHistoryHandler(logging.Handler):
    '''Logging handler using HISTORY FITS cards'''
    def __init__(self, header):
        logging.Handler.__init__(self)
        self.header = header

    def emit(self, record):
        msg = self.format(record)
        self.header.add_history(msg)


def log_to_history(logger, name):
    '''Decorate function, adding a logger handler stored in FITS.'''

    def log_to_history_decorator(method):

        def l2h_method(self, ri):
            history_header = fits.Header()

            fh = FITSHistoryHandler(history_header)
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)

            try:
                result = method(self, ri)
                field = getattr(result, name, None)
                if field:
                    with field.open() as hdulist:
                        hdr = hdulist[0].header
                        hdr.ascardlist().extend(history_header.ascardlist())
                return result
            finally:
                logger.removeHandler(fh)
        return l2h_method

    return log_to_history_decorator
