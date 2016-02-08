#
# Copyright 2015 Universidad Complutense de Madrid
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

'''Unit test for logger'''

import logging 

from  ..logger import FITSHistoryHandler

import pytest
from astropy.io import fits

@pytest.fixture(scope="function")
def logger_fits(request):

    logger = logging.getLogger('numina_test_logger')
    logger.setLevel(logging.DEBUG)
    history_header = fits.Header()
    fh = FITSHistoryHandler(history_header)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    def fin():
        logger.removeHandler(fh)

    request.addfinalizer(fin)

    return logger, history_header  # provide the fixture value


def test_fits_history_handler(logger_fits):

    logger, history_header = logger_fits
    logtext1 = 'Test1'
    logger.info(logtext1)
    logtext2 = 100 * 't'
    logger.info(logtext2)
    hheaders = history_header['HISTORY']

    assert(hheaders[0] == logtext1)
    assert(hheaders[1] == logtext2[:72])
    assert(hheaders[2] == logtext2[72:])

