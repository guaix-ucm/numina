#
# Copyright 2015-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Unit test for logger"""

import logging 

from  ..logger import FITSHistoryHandler, log_to_history

import pytest
from astropy.io import fits

from numina.types.dataframe import DataFrame


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


def test_logger_decorator():

    _logger = logging.getLogger('numina_test_logger')

    class RecipeResult:
        pass

    class Recipe:
        @log_to_history(_logger, "some")
        def decorated_method(self, ri):
            rr = RecipeResult()
            img0 = fits.PrimaryHDU()
            _logger.info("some")
            _logger.info("other")
            img = fits.HDUList([img0])
            some = DataFrame(frame=img)
            rr.some = some
            return rr

    recipe = Recipe()

    rr = recipe.decorated_method(None)
    h = rr.some.open()[0].header
    assert h['history'][0] == 'some'
    assert h['history'][1] == 'other'
