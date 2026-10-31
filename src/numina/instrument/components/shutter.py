#
# Copyright 2016-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from .wheel import Wheel


class Stop:
    def __init__(self, name):
        self.name = name


class Open:
    def __init__(self, name):
        self.name = name


class Shutter(Wheel):
    """Shutter with two positions, OPEN and STOP"""

    def __init__(self, name="Shutter", parent=None):
        super().__init__(name, capacity=2, parent=parent)
        self.put_in_pos(Stop(name="STOP"), 0)
        self.put_in_pos(Open(name="OPEN"), 1)
        self.move_to(1)  # Open by default

    def configure_(self, value):
        # Let's see what is value:
        # a string
        if isinstance(value, str):
            val = value.lower()
            if val == "open":
                val = 1
            elif val == "closed":
                val = 0
            else:
                raise ValueError(f"Not allowed value {value=}")
        elif isinstance(value, int):
            val = value
        else:
            raise TypeError(f"Not allowed type {type(value)}")

        # Move to value
        self.move_to(val)

    def open(self):
        self.move_to(1)

    def close(self):
        self.move_to(0)
