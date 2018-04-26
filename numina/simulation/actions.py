#
# Copyright 2015-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Sequences base class"""


class Sequence(object):
    def __init__(self, instrument, mode):
        self.instrument = instrument
        self.mode = mode

    def setup_instrument(self, instrument):
        pass

    def run(self, **kwds):
        raise NotImplementedError



















