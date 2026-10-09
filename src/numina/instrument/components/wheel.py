#
# Copyright 2016-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Devices with several positions, that hold objects (filters, gratings...)"""

import typing

from typing_extensions import Self

if typing.TYPE_CHECKING:
    from numina.instrument.configorigin import ElementOrigin
    from numina.instrument.hwdevice import DeviceBase


from numina.instrument.hwdevice import HWDevice
from numina.instrument.signal import Signal


class Carrousel(HWDevice):
    """A device with a fixed number of positions, each holding an object.

    The objects are strings or devices with a name. Moving to a position
    selects its object. The signal ``changed`` is emitted when the
    position changes, and ``moved`` after any movement, both with the
    new position.

    Parameters
    ----------
    cid : str
        Name of the device.
    capacity : int
        Number of positions.
    """

    def __init__(
        self,
        cid,
        capacity: int,
        origin: "ElementOrigin | None" = None,
        parent: "DeviceBase | None" = None,
    ):
        super().__init__(name=cid, origin=origin, parent=parent)
        # Container is empty
        self._container = [None] * capacity
        self._capacity = capacity
        self._pos = 0
        # object in the current position
        self._current = self._container[self._pos]

        # signals
        self.changed = Signal()
        self.moved = Signal()

    def current(self):
        """Return the object in the current position"""
        return self._current

    def pos(self):
        """Return the current position"""
        return self._pos

    def put_in_pos(self, obj, pos: int):
        """Put `obj` in the position `pos`"""
        if pos >= self._capacity or pos < 0:
            raise ValueError("position greater than capacity or negative")

        self._container[pos] = obj
        self._current = self._container[self._pos]

    def move_to(self, pos: int):
        """Move to the position `pos`"""
        if pos >= self._capacity or pos < 0:
            raise ValueError(f"Position {pos:d} out of bounds")

        if pos != self._pos:
            self._pos = pos
            self._current = self._container[self._pos]
            self.changed.emit(self._pos)
        self.moved.emit(self._pos)

    def select(self, name):
        """Move to the position of the object named `name`"""
        for idx, item in enumerate(self._container):
            if item:
                if isinstance(item, str):
                    if item == name:
                        return self.move_to(idx)
                elif item.name == name:
                    return self.move_to(idx)
                else:
                    pass
        else:
            raise ValueError(f"No object named {name}")

    @property
    def position(self):
        """The current position, setting it moves the device"""
        return self._pos

    @position.setter
    def position(self, pos: int):
        self.move_to(pos)

    def init_config_info(self):
        """Return the configuration, with the selected object in 'selected'"""
        info = super().init_config_info()
        if self._current:
            if isinstance(self._current, str):
                selected = self._current
            else:
                try:
                    selected = self._current.config_info()
                except AttributeError:
                    selected = self.label
        else:
            selected = self.label
        info["selected"] = selected
        return info

    @property
    def label(self):
        """Name of the object in the current position, 'Unknown' if empty.

        Setting it selects the object with that name.
        """
        if self._current:
            if isinstance(self._current, str):
                lab = self._current
            else:
                lab = self._current.name
        else:
            lab = "Unknown"

        return lab

    @label.setter
    def label(self, name):
        self.select(name)

    @classmethod
    def from_component(
        cls,
        name: str,
        comp_id: str,
        origin: "ElementOrigin | None" = None,
        parent: "DeviceBase | None" = None,
        properties=None,
        setup=None,
    ) -> Self:
        """Create the device from a component, with the 'capacity' of its setup (1 by default)"""
        capacity = 1
        if setup is not None:
            capacity = setup.values["capacity"]

        obj = cls.__new__(cls)
        obj.__init__(comp_id, capacity, origin=origin, parent=parent)
        return obj


class Wheel(Carrousel):
    """A carrousel that can also turn to the next position"""

    def __init__(
        self,
        cid,
        capacity,
        origin: "ElementOrigin | None" = None,
        parent: "DeviceBase | None" = None,
    ):
        super().__init__(cid, capacity, origin=origin, parent=parent)

    def turn(self):
        """Move to the next position, after the last one to the first"""
        self._pos = (self._pos + 1) % self._capacity
        self._current = self._container[self._pos]
        self.changed.emit(self._pos)
        self.moved.emit(self._pos)
