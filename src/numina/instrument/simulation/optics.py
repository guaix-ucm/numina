#
# Copyright 2016-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Simple optical elements, with their transmission"""

import numpy


class Stop:
    """An element that blocks the light"""

    def __init__(self, name):
        self.name = name

    def transmission(self, wl):
        """Transmission at the wavelengths `wl`, always 0"""
        return numpy.zeros_like(wl)


class Open:
    """An empty position, that transmits all the light"""

    def __init__(self, name):
        self.name = name

    def transmission(self, wl):
        """Transmission at the wavelengths `wl`, always 1"""
        return numpy.ones_like(wl)


class Filter:
    """A filter. The argument `transmission` is not used yet"""

    def __init__(self, name, transmission=None):
        self.name = name

    def transmission(self, wl):
        """Transmission at the wavelengths `wl`, 1 for the moment"""
        # FIXME: implement this with a proper
        # transmission
        return numpy.ones_like(wl)
