#
# Copyright 2016-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import numpy

class Stop(object):
    def __init__(self, name):
        self.name = name

    def transmission(self, wl):
        return numpy.zeros_like(wl)


class Open(object):
    def __init__(self, name):
        self.name = name

    def transmission(self, wl):
        return numpy.ones_like(wl)


class Filter(object):
    def __init__(self, name, transmission=None):
        self.name = name

    def transmission(self, wl):
        # FIXME: implement this with a proper
        # transmission
        return numpy.ones_like(wl)
