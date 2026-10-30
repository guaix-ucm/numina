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
    """Object in the closed position of a ``Shutter``."""

    def __init__(self, name):
        self.name = name


class Open:
    """Object in the open position of a ``Shutter``."""

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
        """Move the shutter to the position given by ``value``.

        ``value`` is ``"open"`` or ``"closed"``, in any case, or the index of the
        position, 1 for open and 0 for closed.

        Raises
        ------
        ValueError
            If ``value`` is a string other than ``"open"`` or ``"closed"``.
        TypeError
            If ``value`` is neither a string nor an integer.
        """
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
        """Move the shutter to the open position."""
        self.move_to(1)

    def close(self):
        """Move the shutter to the closed position."""
        self.move_to(0)
