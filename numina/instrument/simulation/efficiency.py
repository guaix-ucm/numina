#
# Copyright 2016-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import numpy


class Efficiency(object):

    def response(self, wl):
        return numpy.ones_like(wl)
