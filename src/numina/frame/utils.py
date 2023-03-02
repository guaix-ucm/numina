#
# Copyright 2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Frame utils"""

import astropy.io.fits as fits


def copy_img(img):
    """Copy an HDUList"""
    return fits.HDUList([hdu.copy() for hdu in img])