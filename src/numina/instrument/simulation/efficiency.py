#
# Copyright 2016-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Efficiency of the elements of an instrument"""

import numpy


class Efficiency:
    """Efficiency as a function of the wavelength, 1 for all of them"""

    def response(self, wl):
        """Efficiency at the wavelengths `wl`"""
        return numpy.ones_like(wl)
