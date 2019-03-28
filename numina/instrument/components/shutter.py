#
# Copyright 2016-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from .wheel import Wheel
from ..simulation.optics import Stop, Open


class Shutter(Wheel):
    """Shutter with two positions, OPEN and CLOSE"""
    def __init__(self, name="Shutter", parent=None):
        super(Shutter, self).__init__(capacity=2, name=name, parent=parent)
        self.put_in_pos(Stop(name='STOP'), 0) # FIXME
        self.put_in_pos(Open(name='OPEN'), 1) # FIXME
        self.move_to(1) # Open by default

    def configure(self, value):
        # Let's see what is value:
        # a string
        if isinstance(value, str):
            val = value.lower()
            if val == 'open':
                val = 1
            elif val == 'closed':
                val = 0
            else:
                raise ValueError('Not allowed value %s', value)
        elif isinstance(value, int):
            val = value
        else:
            raise TypeError('Not allowed type %s', type(value))

        # Move to value
        self.move_to(val)

    def open(self):
        self.move_to(1)

    def close(self):
        self.move_to(0)
